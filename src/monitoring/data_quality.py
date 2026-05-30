from pyspark.sql import DataFrame, functions as F


def check_missing(df: DataFrame, threshold: float = 0.05) -> list[dict]:
    total = df.count()
    results = []
    for col_name in df.columns:
        null_count = df.filter(F.col(col_name).isNull()).count()
        ratio = null_count / total if total > 0 else 1.0
        if ratio > threshold:
            results.append({"column": col_name, "null_ratio": ratio, "alert": True})
    return results


def check_freshness(
    df: DataFrame, timestamp_col: str = "open_time", max_hours: int = 2
) -> dict:
    latest = (
        df.agg(F.max(timestamp_col).alias("max_ts")).collect()[0]["max_ts"]
    )
    if latest is None:
        return {"status": "no_data", "alert": True}
    now = F.current_timestamp()
    age_hours = (now - latest).cast("long") / 3600
    return {"status": "fresh" if age_hours <= max_hours else "stale", "alert": age_hours > max_hours}


def check_schema(df: DataFrame, expected_cols: list[str]) -> list[str]:
    actual = {c.name.lower(): c.name for c in df.schema}
    return [c for c in expected_cols if c.lower() not in actual]
