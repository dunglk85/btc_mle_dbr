# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 00 - Fetch Binance Data To UC Volume

# COMMAND ----------

import csv
import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

# COMMAND ----------

raw_schema = "raw"
volume_name = "landing"
landing_subdir = "btc_hourly"
symbol = "BTCUSDT"
interval = "1h"
default_limit = 24
base_url = "https://data-api.binance.vision/api/v3/klines"
max_page_size = 1000
api_retries = 3
api_retry_sleep_seconds = 2


def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_dev")
limit = int(get_widget("limit", default_limit))
start_date = get_widget("start_date", "")
raw_table_ref = f"{catalog}.{raw_schema}.btc_hourly"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{raw_schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{raw_schema}.{volume_name}")


def latest_raw_open_time_ms():
    try:
        row = spark.sql(f"SELECT max(open_time) AS max_open_time FROM {raw_table_ref}").collect()[0]
        latest = row["max_open_time"]
        if latest is not None:
            return int(latest.replace(tzinfo=timezone.utc).timestamp() * 1000)
    except Exception as exc:
        print(f"latest_raw_open_time_lookup_skipped={exc}")
    return None


start_time = None
if start_date:
    start_time = int(
        datetime.strptime(start_date, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
        * 1000
    )
else:
    latest_open_time_ms = latest_raw_open_time_ms()
    if latest_open_time_ms is not None:
        start_time = latest_open_time_ms + 1
if limit > max_page_size and start_time is None:
    raise ValueError("start_date is required when limit > 1000")

landing_path = f"/Volumes/{catalog}/{raw_schema}/{volume_name}/{landing_subdir}"

print("RUNNING SELF-CONTAINED BINANCE FETCH NOTEBOOK")
print(f"symbol={symbol}")
print(f"interval={interval}")
print(f"limit={limit}")
print(f"start_date={start_date}")
print(f"start_time={start_time}")
print(f"landing_path={landing_path}")

# COMMAND ----------

dbutils.fs.mkdirs(landing_path)

# COMMAND ----------


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
remaining = limit
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
    raise ValueError("No closed Binance klines fetched")

# COMMAND ----------


def format_ts(ms):
    return datetime.fromtimestamp(int(ms) / 1000, timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


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
        raise ValueError("Invalid Binance kline rows before write: " + "; ".join(errors[:20]))


validate_rows(rows)


fetched_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
file_path = f"{landing_path}/btc_hourly_{timestamp}.csv"

with open(file_path, "w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(
        output,
        fieldnames=[
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
            "symbol",
            "interval",
            "fetched_at",
        ],
    )
    writer.writeheader()
    for kline in rows:
        writer.writerow(
            {
                "open_time": format_ts(kline[0]),
                "open": float(kline[1]),
                "high": float(kline[2]),
                "low": float(kline[3]),
                "close": float(kline[4]),
                "volume": float(kline[5]),
                "close_time": format_ts(kline[6]),
                "quote_volume": float(kline[7]),
                "trades": int(kline[8]),
                "source": "binance",
                "symbol": symbol,
                "interval": interval,
                "fetched_at": fetched_at,
            }
        )

print(f"uploaded_file={file_path}")

# COMMAND ----------

display(spark.read.option("header", True).csv(file_path).limit(10))
