import os
import time
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
    retries: int = 3,
    retry_sleep_seconds: int = 2,
) -> list:
    client = get_binance_client()
    return fetch_klines_with_client(
        client,
        symbol=symbol,
        interval=interval,
        limit=limit,
        start_time=start_time,
        end_time=end_time,
        retries=retries,
        retry_sleep_seconds=retry_sleep_seconds,
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
    retries: int = 3,
    retry_sleep_seconds: int = 2,
) -> list:
    all_data = []
    remaining = limit
    current_start = start_time
    while remaining > 0:
        page_size = min(remaining, PAGE_SIZE)
        for attempt in range(1, retries + 1):
            try:
                page = client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=page_size,
                    startTime=current_start,
                    endTime=end_time,
                )
                break
            except Exception:
                if attempt == retries:
                    raise
                time.sleep(retry_sleep_seconds * attempt)
        if not page:
            break
        all_data.extend(page)
        remaining -= len(page)
        if len(page) < page_size:
            break
        current_start = int(page[-1][0]) + 1
    return all_data


def validate_klines(raw: list) -> None:
    seen_open_times = set()
    errors = []
    for idx, row in enumerate(raw):
        try:
            open_time = int(row[0])
            close_time = int(row[6])
            open_price = float(row[1])
            high = float(row[2])
            low = float(row[3])
            close = float(row[4])
            volume = float(row[5])
            quote_volume = float(row[7])
            trades = int(row[8])
        except Exception as exc:
            errors.append(f"row={idx} parse_error={exc}")
            continue

        if open_time in seen_open_times:
            errors.append(f"row={idx} duplicate_open_time={open_time}")
        seen_open_times.add(open_time)
        if close_time <= open_time:
            errors.append(f"row={idx} close_time_not_after_open_time")
        if min(open_price, high, low, close) <= 0:
            errors.append(f"row={idx} non_positive_price")
        if high < max(open_price, low, close):
            errors.append(f"row={idx} high_below_ohlc")
        if low > min(open_price, high, close):
            errors.append(f"row={idx} low_above_ohlc")
        if volume < 0 or quote_volume < 0 or trades < 0:
            errors.append(f"row={idx} negative_volume_or_trades")

    if errors:
        raise ValueError("Invalid Binance kline rows before write: " + "; ".join(errors[:20]))


def klines_to_rows(raw: list, symbol: str = "BTCUSDT", interval: str = "1h") -> list:
    validate_klines(raw)
    fetched_at = datetime.now(timezone.utc)
    return [
        {
            "open_time": datetime.fromtimestamp(k[0] / 1000, timezone.utc),
            "open": float(k[1]),
            "high": float(k[2]),
            "low": float(k[3]),
            "close": float(k[4]),
            "volume": float(k[5]),
            "close_time": datetime.fromtimestamp(k[6] / 1000, timezone.utc),
            "quote_volume": float(k[7]),
            "trades": int(k[8]),
            "source": "binance",
            "symbol": symbol,
            "interval": interval,
            "fetched_at": fetched_at,
            "ingested_at": fetched_at,
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
            StructField("symbol", StringType(), True),
            StructField("interval", StringType(), True),
            StructField("fetched_at", TimestampType(), True),
            StructField("ingested_at", TimestampType(), True),
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
    catalog: str = "btc_simply",
    schema: str = "raw",
    table: str = "btc_hourly",
    backfill_start_ms: int = DEFAULT_BACKFILL_START_MS,
    symbol: str = "BTCUSDT",
    interval: str = "1h",
) -> DataFrame:
    table_ref = f"{catalog}.{schema}.{table}"
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    exists = table_exists(spark, table_ref)
    last_ts = get_latest_timestamp(spark, table_ref) if exists else None
    if last_ts:
        start_ts = last_ts + 1
        raw = fetch_klines(symbol=symbol, interval=interval, start_time=start_ts, limit=100000)
    else:
        raw = fetch_klines(symbol=symbol, interval=interval, start_time=backfill_start_ms, limit=100000)
    if not raw:
        if not exists:
            empty_df = spark.createDataFrame([], schema=SCHEMA)
            empty_df.write.format("delta").mode("overwrite").saveAsTable(table_ref)
        return spark.table(table_ref)
    rows = klines_to_rows(raw, symbol=symbol, interval=interval)
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
