from datetime import datetime
from pyspark.sql import DataFrame, SparkSession, functions as F


def check_missing(df: DataFrame, threshold: float = 0.05) -> list[dict]:
    total = df.count()
    if total == 0:
        return [{"column": col, "null_ratio": 1.0, "alert": True} for col in df.columns]
    null_counts = df.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0]
    results = []
    for col_name in df.columns:
        ratio = null_counts[col_name] / total
        if ratio > threshold:
            results.append({"column": col_name, "null_ratio": ratio, "alert": True})
    return results


def check_freshness(
    spark: SparkSession, df: DataFrame, timestamp_col: str = "open_time", max_hours: int = 2
) -> dict:
    latest = (
        df.agg(F.max(timestamp_col).alias("max_ts")).collect()[0]["max_ts"]
    )
    if latest is None:
        return {"status": "no_data", "alert": True}
    now = datetime.now(tz=latest.tzinfo if hasattr(latest, 'tzinfo') else None)
    age_hours = (now - latest).total_seconds() / 3600
    return {"status": "fresh" if age_hours <= max_hours else "stale", "alert": age_hours > max_hours}


def check_schema(df: DataFrame, expected_cols: list[str]) -> list[str]:
    actual = {c.name.lower(): c.name for c in df.schema}
    return [c for c in expected_cols if c.lower() not in actual]
