# Databricks notebook source
# MAGIC %md # 01 - Data Ingestion

from pyspark.sql import functions as F

catalog = "btc_dev"
raw_schema = "raw"
volume_name = "landing"
table_name = "btc_hourly"
landing_path = f"/Volumes/{catalog}/{raw_schema}/{volume_name}/btc_hourly"
table_ref = f"{catalog}.{raw_schema}.{table_name}"

print("RUNNING SELF-CONTAINED INGESTION NOTEBOOK")
print(f"landing_path={landing_path}")
print(f"table_ref={table_ref}")

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

raw = spark.read.option("header", True).csv(landing_path)
raw_count = raw.count()
print(f"raw_landing_count={raw_count}")
if raw_count == 0:
    raise ValueError(f"No CSV rows found at {landing_path}")

# Accept both new Spark-friendly timestamps and older ISO timestamps already in Volume.
parsed = raw.select(
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
)

null_open_time_count = parsed.filter(F.col("open_time").isNull()).count()
print(f"null_open_time_count={null_open_time_count}")
if null_open_time_count > 0:
    display(raw.filter(F.col("open_time").isNotNull()).select("open_time", "close_time").limit(20))
    raise ValueError(f"Found {null_open_time_count} rows with unparseable open_time")

deduped = parsed.dropDuplicates(["open_time"])
deduped_count = deduped.count()
print(f"deduped_landing_count={deduped_count}")
if deduped_count == 0:
    raise ValueError(f"No parsed rows available from {landing_path}")

deduped.createOrReplaceTempView("_btc_hourly_landing")

spark.sql(f"""
    MERGE INTO {table_ref} AS target
    USING _btc_hourly_landing AS source
    ON target.open_time = source.open_time
    WHEN MATCHED THEN UPDATE SET *
    WHEN NOT MATCHED THEN INSERT *
""")

result = spark.table(table_ref)
print(f"table_count_after_merge={result.count()}")
display(result.orderBy("open_time").limit(10))
display(result.orderBy(F.col("open_time").desc()).limit(10))
