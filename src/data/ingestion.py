import requests
from datetime import datetime, timedelta
from typing import Optional
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, DoubleType, TimestampType


BINANCE_BASE = "https://api.binance.com/api/v3"


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 500,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
) -> list:
    params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1000)}
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time
    resp = requests.get(f"{BINANCE_BASE}/klines", params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


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
        StructField("trades", DoubleType(), True),
    ]
)


def get_latest_timestamp(spark: SparkSession, table: str) -> Optional[int]:
    try:
        last = (
            spark.sql(f"SELECT max(open_time) AS max_ot FROM {table}")
            .collect()[0]["max_ot"]
        )
        if last:
            return int(last.timestamp() * 1000)
    except Exception:
        pass
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
        raw = fetch_klines(start_time=start_ts)
    else:
        raw = fetch_klines(limit=1000)
    if not raw:
        return spark.table(table_ref)
    rows = klines_to_rows(raw)
    df = spark.createDataFrame(rows, schema=SCHEMA)
    df.write.mode("append").saveAsTable(table_ref)
    return spark.table(table_ref)
