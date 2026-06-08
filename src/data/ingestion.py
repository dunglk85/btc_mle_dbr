import os
from datetime import datetime, timezone
from typing import Any, Optional

try:
    from pyspark.sql import SparkSession, DataFrame, Window, functions as F
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
    Window = None
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
            StructField("open_time", StringType(), True),
            StructField("open", DoubleType(), True),
            StructField("high", DoubleType(), True),
            StructField("low", DoubleType(), True),
            StructField("close", DoubleType(), True),
            StructField("volume", DoubleType(), True),
            StructField("close_time", StringType(), True),
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
    checkpoint_subdir: str = "_checkpoints/btc_hourly",
    schema_subdir: str = "_schemas/btc_hourly",
    staging_retention_hours: int = 48,
) -> DataFrame:
    run_started_at = datetime.now(timezone.utc)
    table_ref = f"{catalog}.{raw_schema}.{table}"
    staging_table_ref = f"{catalog}.{raw_schema}.{table}_landing_autoloader"
    landing_path = f"/Volumes/{catalog}/{raw_schema}/{volume_name}/{landing_subdir}"
    checkpoint_path = f"/Volumes/{catalog}/{raw_schema}/{volume_name}/{checkpoint_subdir}"
    schema_path = f"/Volumes/{catalog}/{raw_schema}/{volume_name}/{schema_subdir}"
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
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {staging_table_ref} (
            open_time STRING,
            open DOUBLE,
            high DOUBLE,
            low DOUBLE,
            close DOUBLE,
            volume DOUBLE,
            close_time STRING,
            quote_volume DOUBLE,
            trades BIGINT,
            source STRING,
            _source_file STRING,
            _loaded_at TIMESTAMP
        )
        USING DELTA
    """)

    print(f"load_landing_to_raw: landing_path={landing_path}")
    print(f"load_landing_to_raw: checkpoint_path={checkpoint_path}")
    stream_df = (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "csv")
        .option("cloudFiles.schemaLocation", schema_path)
        .option("header", True)
        .schema(LANDING_SCHEMA)
        .load(landing_path)
        .select(
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_volume",
            "trades",
            "source",
            F.col("_metadata.file_path").alias("_source_file"),
            F.current_timestamp().alias("_loaded_at"),
        )
    )

    query = (
        stream_df.writeStream.option("checkpointLocation", checkpoint_path)
        .outputMode("append")
        .trigger(availableNow=True)
        .toTable(staging_table_ref)
    )
    query.awaitTermination()
    merge_landing_staging_to_raw(
        spark,
        staging_table_ref,
        table_ref,
        landing_path,
        staging_retention_hours=staging_retention_hours,
        run_started_at=run_started_at,
    )
    return spark.table(table_ref)


def merge_landing_staging_to_raw(
    spark: SparkSession,
    staging_table_ref: str,
    table_ref: str,
    landing_path: str,
    staging_retention_hours: int = 48,
    run_started_at: Optional[datetime] = None,
) -> None:
    if run_started_at is None:
        raise ValueError("run_started_at is required to avoid merging retained staging rows")

    raw = spark.table(staging_table_ref)
    raw = raw.filter(F.col("_loaded_at") >= F.lit(run_started_at))
    raw_landing_count = raw.count()
    print(f"merge_landing_staging_to_raw: raw_landing_count={raw_landing_count}")
    if raw_landing_count == 0:
        print("merge_landing_staging_to_raw: empty staging table, skipping merge")
        return

    df = raw.select(
        F.coalesce(
            F.to_timestamp("open_time", "yyyy-MM-dd HH:mm:ss"),
            F.to_timestamp("open_time", "yyyy-MM-dd'T'HH:mm:ssXXX"),
            F.to_timestamp("open_time"),
        ).alias("open_time"),
        F.col("open").cast("double").alias("open"),
        F.col("high").cast("double").alias("high"),
        F.col("low").cast("double").alias("low"),
        F.col("close").cast("double").alias("close"),
        F.col("volume").cast("double").alias("volume"),
        F.coalesce(
            F.to_timestamp("close_time", "yyyy-MM-dd HH:mm:ss"),
            F.to_timestamp("close_time", "yyyy-MM-dd'T'HH:mm:ssXXX"),
            F.to_timestamp("close_time"),
        ).alias("close_time"),
        F.col("quote_volume").cast("double").alias("quote_volume"),
        F.col("trades").cast("bigint").alias("trades"),
        F.coalesce(F.col("source"), F.lit("binance")).alias("source"),
        F.current_timestamp().alias("ingested_at"),
        F.col("_source_file"),
        F.col("_loaded_at"),
    )
    null_open_time_count = df.filter(F.col("open_time").isNull()).count()
    print(f"merge_landing_staging_to_raw: null_open_time_count={null_open_time_count}")
    if null_open_time_count > 0:
        raise ValueError(
            f"Found {null_open_time_count} landing rows with unparseable open_time "
            f"at {landing_path}"
        )

    dedupe_window = Window.partitionBy("open_time").orderBy(
        F.col("_loaded_at").desc(),
        F.col("_source_file").desc(),
    )
    df = (
        df.withColumn("_row_number", F.row_number().over(dedupe_window))
        .filter(F.col("_row_number") == 1)
        .drop("_row_number", "_source_file", "_loaded_at")
    )
    landing_count = df.count()
    print(f"merge_landing_staging_to_raw: parsed_distinct_landing_count={landing_count}")
    if landing_count == 0:
        raise ValueError(
            f"No parsed landing rows found at {landing_path}; "
            f"raw_landing_count={raw_landing_count}"
        )
    df.createOrReplaceTempView("_btc_hourly_landing")

    spark.sql(f"""
        MERGE INTO {table_ref} AS target
        USING _btc_hourly_landing AS source
        ON target.open_time = source.open_time
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)

    spark.sql(f"""
        DELETE FROM {staging_table_ref}
        WHERE _loaded_at < current_timestamp() - INTERVAL {staging_retention_hours} HOURS
    """)
    print(
        "merge_landing_staging_to_raw: "
        f"staging_rows_after_retention_cleanup={spark.table(staging_table_ref).count()}"
    )
