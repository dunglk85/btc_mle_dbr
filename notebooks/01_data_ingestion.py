# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 - Data Ingestion

# COMMAND ----------

from pyspark.sql import Window, functions as F
from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_dev")
raw_schema = "raw"
volume_name = "landing"
table_name = "btc_hourly"
landing_path = f"/Volumes/{catalog}/{raw_schema}/{volume_name}/btc_hourly"
checkpoint_path = f"/Volumes/{catalog}/{raw_schema}/{volume_name}/_checkpoints/btc_hourly"
schema_path = f"/Volumes/{catalog}/{raw_schema}/{volume_name}/_schemas/btc_hourly"
table_ref = f"{catalog}.{raw_schema}.{table_name}"
staging_table_ref = f"{catalog}.{raw_schema}.{table_name}_landing_autoloader"
staging_retention_hours = int(get_widget("staging_retention_hours", "48"))

print("RUNNING SELF-CONTAINED AUTO LOADER INGESTION NOTEBOOK")
print(f"landing_path={landing_path}")
print(f"checkpoint_path={checkpoint_path}")
print(f"table_ref={table_ref}")
print(f"staging_table_ref={staging_table_ref}")
print(f"staging_retention_hours={staging_retention_hours}")

# COMMAND ----------

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

landing_schema = StructType(
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

# COMMAND ----------

def parse_and_merge_staging():
    raw = spark.table(staging_table_ref)
    raw_count = raw.count()
    print(f"staging_landing_count={raw_count}")
    if raw_count == 0:
        print("empty Auto Loader staging table; skipping merge")
        return

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
        F.col("_source_file"),
    )

    null_open_time_count = parsed.filter(F.col("open_time").isNull()).count()
    print(f"null_open_time_count={null_open_time_count}")
    if null_open_time_count > 0:
        display(raw.filter(F.col("open_time").isNotNull()).select("open_time", "close_time").limit(20))
        raise ValueError(f"Found {null_open_time_count} rows with unparseable open_time")

    dedupe_window = Window.partitionBy("open_time").orderBy(F.col("_source_file").desc())
    deduped = (
        parsed.withColumn("_row_number", F.row_number().over(dedupe_window))
        .filter(F.col("_row_number") == 1)
        .drop("_row_number", "_source_file")
    )
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

    spark.sql(f"""
        DELETE FROM {staging_table_ref}
        WHERE _loaded_at < current_timestamp() - INTERVAL {staging_retention_hours} HOURS
    """)
    print(f"staging_rows_after_retention_cleanup={spark.table(staging_table_ref).count()}")

# COMMAND ----------

stream_df = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "csv")
    .option("cloudFiles.schemaLocation", schema_path)
    .option("header", True)
    .schema(landing_schema)
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

# COMMAND ----------

query = (
    stream_df.writeStream.option("checkpointLocation", checkpoint_path)
    .outputMode("append")
    .trigger(availableNow=True)
    .toTable(staging_table_ref)
)
query.awaitTermination()

parse_and_merge_staging()

result = spark.table(table_ref)
print(f"table_count_after_merge={result.count()}")

# COMMAND ----------

display(result.orderBy("open_time").limit(10))

# COMMAND ----------

display(result.orderBy(F.col("open_time").desc()).limit(10))
