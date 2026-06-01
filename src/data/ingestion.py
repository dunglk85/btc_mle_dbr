import os
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from pyspark.sql import SparkSession, DataFrame, functions as F
    from pyspark.sql.types import (
        StructType,
        StructField,
        DoubleType,
        LongType,
        StringType,
        TimestampType,
    )
except ImportError:
    SparkSession = Any
    DataFrame = Any
    F = None
    StructType = StructField = DoubleType = LongType = StringType = TimestampType = None

from src.utils.logger import get_logger

logger = get_logger(__name__)


PAGE_SIZE = 1000
DEFAULT_BACKFILL_START_MS = int(
    datetime(2025, 1, 1, tzinfo=timezone.utc).timestamp() * 1000
)


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> list:
    client = get_binance_client()
    return fetch_klines_with_client(
        client,
        symbol=symbol,
        interval=interval,
        limit=limit,
        start_time=start_time,
        end_time=end_time,
    )


def get_binance_client():
    try:
        from binance.client import Client
    except ImportError as e:
        raise RuntimeError(
            "python-binance is required for Binance ingestion. "
            "Install it with `pip install python-binance`."
        ) from e

    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")
    if api_key and api_secret:
        return Client(api_key, api_secret)

    return Client()


def fetch_klines_with_client(
    client,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> list:
    all_data = []
    remaining = limit
    current_start = start_time
    while remaining > 0:
        page_size = min(remaining, PAGE_SIZE)
        page = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=page_size,
            startTime=current_start,
            endTime=end_time,
        )
        if not page:
            break
        all_data.extend(page)
        remaining -= len(page)
        if len(page) < page_size:
            break
        current_start = int(page[-1][0]) + 1
    return all_data


def klines_to_rows(raw: list) -> list:
    return [
        {
            "open_time": datetime.fromtimestamp(k[0] / 1000),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "close_time": datetime.fromtimestamp(k[6] / 1000),
            "quote_volume": float(k[7]),
            "trades": int(k[8]),
        }
        for k in raw
    ]


SCHEMA = None
if StructType is not None:
    SCHEMA = StructType(
        [
            StructField("open_time", TimestampType(), True),
            StructField("open", DoubleType(), True),
            StructField("high", DoubleType(), True),
            StructField("low", DoubleType(), True),
            StructField("close", DoubleType(), True),
            StructField("volume", DoubleType(), True),
            StructField("close_time", TimestampType(), True),
            StructField("quote_volume", DoubleType(), True),
            StructField("trades", LongType(), True),
            StructField("source", StringType(), True),
            StructField("ingested_at", TimestampType(), True),
        ]
    )

LANDING_SCHEMA = None
if StructType is not None:
    LANDING_SCHEMA = StructType(
        [
            StructField("open_time", TimestampType(), True),
            StructField("open", DoubleType(), True),
            StructField("high", DoubleType(), True),
            StructField("low", DoubleType(), True),
            StructField("close", DoubleType(), True),
            StructField("volume", DoubleType(), True),
            StructField("close_time", TimestampType(), True),
            StructField("quote_volume", DoubleType(), True),
            StructField("trades", LongType(), True),
            StructField("source", StringType(), True),
        ]
    )


def get_latest_timestamp(spark: SparkSession, table: str) -> Optional[int]:
    try:
        last = (
            spark.table(table).agg(F.max("open_time").alias("max_ot"))
            .collect()[0]["max_ot"]
        )
        if last:
            return int(last.timestamp() * 1000)
    except Exception as e:
        logger.warning("get_latest_timestamp failed", table=table, error=str(e))
    return None


def table_exists(spark: SparkSession, table: str) -> bool:
    try:
        spark.table(table).limit(1).collect()
        return True
    except Exception:
        return False


def incremental_ingest(
    spark: SparkSession,
    catalog: str = "btc_dev",
    schema: str = "raw",
    table: str = "btc_hourly",
    backfill_start_ms: int = DEFAULT_BACKFILL_START_MS,
) -> DataFrame:
    table_ref = f"{catalog}.{schema}.{table}"
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    exists = table_exists(spark, table_ref)
    last_ts = get_latest_timestamp(spark, table_ref) if exists else None
    if last_ts:
        start_ts = last_ts + 1
        raw = fetch_klines(start_time=start_ts, limit=100000)
    else:
        raw = fetch_klines(start_time=backfill_start_ms, limit=100000)
    if not raw:
        if not exists:
            empty_df = spark.createDataFrame([], schema=SCHEMA)
            empty_df.write.format("delta").mode("overwrite").saveAsTable(table_ref)
        return spark.table(table_ref)
    rows = klines_to_rows(raw)
    df = spark.createDataFrame(rows, schema=SCHEMA)
    if not exists:
        df.write.format("delta").mode("overwrite").saveAsTable(table_ref)
        return spark.table(table_ref)
    df.createOrReplaceTempView("_new_data")
    spark.sql(f"""
        MERGE INTO {table_ref} AS target
        USING _new_data AS source
        ON target.open_time = source.open_time
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    return spark.table(table_ref)


def load_landing_to_raw(
    spark: SparkSession,
    catalog: str = "btc_dev",
    raw_schema: str = "raw",
    volume_name: str = "landing",
    table: str = "btc_hourly",
    landing_subdir: str = "btc_hourly",
) -> DataFrame:
    table_ref = f"{catalog}.{raw_schema}.{table}"
    landing_path = f"/Volumes/{catalog}/{raw_schema}/{volume_name}/{landing_subdir}"
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{raw_schema}")
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{raw_schema}.{volume_name}")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {table_ref} (
            open_time TIMESTAMP,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            close_time TIMESTAMP,
            quote_volume DOUBLE,
            trades BIGINT,
            source STRING,
            ingested_at TIMESTAMP
        )
        USING DELTA
    """)

    df = spark.read.option("header", True).schema(LANDING_SCHEMA).csv(landing_path)
    df = df.withColumn("source", F.coalesce(F.col("source"), F.lit("binance")))
    df = df.withColumn("ingested_at", F.current_timestamp())
    df = df.dropDuplicates(["open_time"])
    landing_count = df.count()
    if landing_count == 0:
        raise ValueError(f"No landing rows found at {landing_path}")
    df.createOrReplaceTempView("_btc_hourly_landing")

    spark.sql(f"""
        MERGE INTO {table_ref} AS target
        USING _btc_hourly_landing AS source
        ON target.open_time = source.open_time
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    return spark.table(table_ref)
