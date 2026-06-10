from pyspark.sql import DataFrame, functions as F, Window


def compute_features(
    df: DataFrame,
    ma_windows: list[int] | None = None,
    lag_hours: list[int] | None = None,
) -> DataFrame:
    if ma_windows is None:
        ma_windows = [7, 24, 168]
    if lag_hours is None:
        lag_hours = [1, 2, 4, 12, 24]

    df = df.dropDuplicates(["open_time"]).orderBy("open_time")
    w = Window.orderBy("open_time")

    for w_size in ma_windows:
        df = df.withColumn(f"ma_{w_size}", F.avg("close").over(w.rowsBetween(-w_size, -1)))

    for h in lag_hours:
        df = df.withColumn(f"close_lag_{h}h", F.lag("close", h).over(w))

    df = df.withColumn("return_1h", (F.col("close") / F.lag("close", 1).over(w)) - F.lit(1.0))
    df = df.withColumn("hl_spread", F.col("high") - F.col("low"))
    df = df.withColumn("oc_change", F.col("close") - F.col("open"))
    df = df.withColumn("hour", F.hour("open_time"))
    df = df.withColumn("day_of_week", F.dayofweek("open_time"))
    target = df.select(
        (F.col("open_time") - F.expr("INTERVAL 1 HOUR")).alias("open_time"),
        F.col("close").alias("target_close_1h"),
    )
    df = df.join(target, on="open_time", how="left")

    return df


def write_features_table(
    spark,
    catalog: str = "btc_simply",
    raw_schema: str = "raw",
    features_schema: str = "features",
    raw_table: str = "btc_hourly",
    features_table: str = "btc_features",
) -> DataFrame:
    raw_ref = f"{catalog}.{raw_schema}.{raw_table}"
    features_ref = f"{catalog}.{features_schema}.{features_table}"
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{features_schema}")
    raw_df = spark.table(raw_ref)
    features_df = compute_features(raw_df)
    features_df.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(features_ref)
    return spark.table(features_ref)
