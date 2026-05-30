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

    w = Window.orderBy("open_time")

    for w_size in ma_windows:
        df = df.withColumn(f"ma_{w_size}", F.avg("close").over(w.rowsBetween(-w_size, -1)))

    for h in lag_hours:
        df = df.withColumn(f"close_lag_{h}h", F.lag("close", h).over(w))

    df = df.withColumn("close_time", F.col("open_time"))
    df = df.withColumn("hour", F.hour("open_time"))
    df = df.withColumn("day_of_week", F.dayofweek("open_time"))

    return df
