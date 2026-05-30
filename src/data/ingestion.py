import requests
from datetime import datetime, timedelta
from typing import Optional
from pyspark.sql import SparkSession, DataFrame, functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, LongType, TimestampType

from src.utils.logger import get_logger

BINANCE_BASE = "https://api.binance.com/api/v3"
logger = get_logger(__name__)


PAGE_SIZE = 1000


def fetch_klines(
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
        params = {"symbol": symbol, "interval": interval, "limit": page_size}
        if current_start is not None:
            params["startTime"] = current_start
        if end_time:
            params["endTime"] = end_time
        resp = requests.get(f"{BINANCE_BASE}/klines", params=params, timeout=30)
        resp.raise_for_status()
        page = resp.json()
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


def incremental_ingest(
    spark: SparkSession,
    catalog: str = "btc_dev",
    schema: str = "raw",
    table: str = "btc_hourly",
) -> DataFrame:
    table_ref = f"{catalog}.{schema}.{table}"
    last_ts = get_latest_timestamp(spark, table_ref)
    if last_ts:
        start_ts = last_ts + 1
        raw = fetch_klines(start_time=start_ts, limit=100000)
    else:
        raw = fetch_klines(limit=100000)
    if not raw:
        return spark.table(table_ref)
    rows = klines_to_rows(raw)
    df = spark.createDataFrame(rows, schema=SCHEMA)
    df.createOrReplaceTempView("_new_data")
    spark.sql(f"""
        MERGE INTO {table_ref} AS target
        USING _new_data AS source
        ON target.open_time = source.open_time
        WHEN MATCHED THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
    """)
    return spark.table(table_ref)
