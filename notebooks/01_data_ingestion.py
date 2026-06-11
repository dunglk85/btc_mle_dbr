# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 - Data Ingestion
# MAGIC Fetch closed BTC hourly candles from Binance and merge directly into the raw Delta table.

# COMMAND ----------

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from pyspark.sql import Window, functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType, TimestampType

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_simply")
raw_schema = "raw"
table_name = "btc_hourly"
table_ref = f"{catalog}.{raw_schema}.{table_name}"
symbol = "BTCUSDT"
interval = "1h"
requested_limit = int(get_widget("limit", "24"))
start_date = get_widget("start_date", "")
default_backfill_start_date = get_widget("default_backfill_start_date", "2025-01-01")
backfill_limit = int(get_widget("backfill_limit", "1000000"))
base_url = "https://data-api.binance.vision/api/v3/klines"
max_page_size = 1000
api_retries = 3
api_retry_sleep_seconds = 2
run_started_at = datetime.now(timezone.utc)

print("RUNNING DIRECT BINANCE INGESTION NOTEBOOK")
print(f"table_ref={table_ref}")
print(f"symbol={symbol}")
print(f"interval={interval}")
print(f"requested_limit={requested_limit}")
print(f"start_date={start_date}")
print(f"default_backfill_start_date={default_backfill_start_date}")
print(f"backfill_limit={backfill_limit}")
print(f"run_started_at={run_started_at.isoformat()}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{raw_schema}")
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
        symbol STRING,
        interval STRING,
        fetched_at TIMESTAMP,
        ingested_at TIMESTAMP
    )
    USING DELTA
""")
for column_def in ["symbol STRING", "interval STRING", "fetched_at TIMESTAMP"]:
    try:
        spark.sql(f"ALTER TABLE {table_ref} ADD COLUMNS ({column_def})")
    except Exception as exc:
        print(f"raw_column_already_exists_or_add_skipped={column_def}: {exc}")

# COMMAND ----------

def latest_raw_open_time_ms():
    try:
        row = spark.sql(f"SELECT max(open_time) AS max_open_time FROM {table_ref}").collect()[0]
        latest = row["max_open_time"]
        if latest is not None:
            return int(latest.replace(tzinfo=timezone.utc).timestamp() * 1000)
    except Exception as exc:
        print(f"latest_raw_open_time_lookup_skipped={exc}")
    return None


if start_date:
    start_time = int(
        datetime.strptime(start_date, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )
    effective_limit = backfill_limit if requested_limit == 24 else requested_limit
    print("mode=backfill_from_start_date")
else:
    latest_open_time_ms = latest_raw_open_time_ms()
    if latest_open_time_ms is not None:
        start_time = latest_open_time_ms + 1
        effective_limit = requested_limit
        print("mode=incremental")
    else:
        start_time = int(
            datetime.strptime(default_backfill_start_date, "%Y-%m-%d")
            .replace(tzinfo=timezone.utc)
            .timestamp()
            * 1000
        )
        effective_limit = backfill_limit
        print("mode=initial_backfill_from_default_start_date")

print(f"start_time={start_time}")
print(f"effective_limit={effective_limit}")


def fetch_page(page_limit, page_start_time=None):
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": page_limit,
    }
    if page_start_time is not None:
        params["startTime"] = page_start_time

    url = f"{base_url}?{urllib.parse.urlencode(params)}"
    for attempt in range(1, api_retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception:
            if attempt == api_retries:
                raise
            sleep_seconds = api_retry_sleep_seconds * attempt
            print(f"binance_fetch_retry attempt={attempt} sleep_seconds={sleep_seconds}")
            time.sleep(sleep_seconds)


rows = []
remaining = effective_limit
current_start = start_time
while remaining > 0:
    page_limit = min(remaining, max_page_size)
    page = fetch_page(page_limit, current_start)
    if not page:
        break
    rows.extend(page)
    remaining -= len(page)
    if len(page) < page_limit:
        break
    current_start = int(page[-1][0]) + 1

print(f"fetched_rows={len(rows)}")
if not rows:
    raise ValueError("No Binance klines fetched")

now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
closed_rows = [row for row in rows if int(row[6]) < now_ms]
filtered_count = len(rows) - len(closed_rows)
print(f"filtered_open_candles={filtered_count}")
rows = closed_rows
if not rows:
    print("no_new_closed_klines=skip")
    dbutils.notebook.exit("SKIPPED: No closed klines to ingest")

# COMMAND ----------

def validate_rows(klines):
    seen_open_times = set()
    errors = []
    for idx, row in enumerate(klines):
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
        raise ValueError("Invalid Binance kline rows before merge: " + "; ".join(errors[:20]))


def to_ts(ms):
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc)


validate_rows(rows)
fetched_at = datetime.now(timezone.utc)
records = [
    {
        "open_time": to_ts(kline[0]),
        "open": float(kline[1]),
        "high": float(kline[2]),
        "low": float(kline[3]),
        "close": float(kline[4]),
        "volume": float(kline[5]),
        "close_time": to_ts(kline[6]),
        "quote_volume": float(kline[7]),
        "trades": int(kline[8]),
        "source": "binance",
        "symbol": symbol,
        "interval": interval,
        "fetched_at": fetched_at,
        "ingested_at": fetched_at,
    }
    for kline in rows
]

raw_schema_struct = StructType(
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

incoming = spark.createDataFrame(records, schema=raw_schema_struct)
dedupe_window = Window.partitionBy("open_time").orderBy(F.col("fetched_at").desc())
deduped = (
    incoming.withColumn("_row_number", F.row_number().over(dedupe_window))
    .filter(F.col("_row_number") == 1)
    .drop("_row_number")
)
deduped_count = deduped.count()
print(f"deduped_binance_count={deduped_count}")
if deduped_count == 0:
    print("no_deduped_rows=skip")
    dbutils.notebook.exit("SKIPPED: All rows are duplicates or empty")

deduped.createOrReplaceTempView("_btc_hourly_binance")
spark.sql(f"""
    MERGE INTO {table_ref} AS target
    USING _btc_hourly_binance AS source
    ON target.open_time = source.open_time
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

result = spark.table(table_ref)
print(f"table_count_after_merge={result.count()}")

# COMMAND ----------

display(result.orderBy("open_time").limit(10))

# COMMAND ----------

display(result.orderBy(F.col("open_time").desc()).limit(10))
