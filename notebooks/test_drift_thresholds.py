# Databricks notebook source

# MAGIC %md
# MAGIC # Test Drift Thresholds
# MAGIC Validate PSI/KS drift thresholds against historical BTC data.

# COMMAND ----------

from datetime import datetime, timedelta, timezone
import math

from pyspark.sql import functions as F

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_simply")
features_ref = f"{catalog}.features.btc_features"

# Test different threshold combinations
psi_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 1.0]
ks_thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]

# Time windows to test (hours)
window_configs = [
    {"recent": 24, "reference": 168, "label": "24h vs 7d"},
    {"recent": 48, "reference": 336, "label": "48h vs 14d"},
    {"recent": 168, "reference": 720, "label": "168h vs 30d"},
    {"recent": 336, "reference": 1440, "label": "336h vs 60d"},
]

print(f"features_ref={features_ref}")

# COMMAND ----------

# --- PSI and KS functions (copy from 06_monitoring.py) ---

def approx_ks(reference_df, recent_df, col_name, probs=None):
    if probs is None:
        probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    quantiles = reference_df.approxQuantile(col_name, probs, 0.01)
    if not quantiles:
        return None
    ref_count = reference_df.count()
    recent_count = recent_df.count()
    if ref_count == 0 or recent_count == 0:
        return None
    max_delta = 0.0
    for quantile in quantiles:
        ref_cdf = reference_df.filter(F.col(col_name) <= quantile).count() / ref_count
        recent_cdf = recent_df.filter(F.col(col_name) <= quantile).count() / recent_count
        max_delta = max(max_delta, abs(ref_cdf - recent_cdf))
    return max_delta


def psi(reference_df, recent_df, col_name, buckets=10):
    ref_count = reference_df.count()
    recent_count = recent_df.count()
    if ref_count == 0 or recent_count == 0:
        return None
    probs = [i / buckets for i in range(1, buckets)]
    cuts = reference_df.approxQuantile(col_name, probs, 0.01)
    if not cuts:
        return None
    boundaries = [None, *cuts, None]
    total = 0.0
    epsilon = 1e-6
    for idx in range(len(boundaries) - 1):
        lower = boundaries[idx]
        upper = boundaries[idx + 1]
        ref_bucket = reference_df.filter(F.col(col_name) > lower) if lower is not None else reference_df
        recent_bucket = recent_df.filter(F.col(col_name) > lower) if lower is not None else recent_df
        if upper is not None:
            ref_bucket = ref_bucket.filter(F.col(col_name) <= upper)
            recent_bucket = recent_bucket.filter(F.col(col_name) <= upper)
        ref_pct = max(ref_bucket.count() / ref_count, epsilon)
        recent_pct = max(recent_bucket.count() / recent_count, epsilon)
        total += (recent_pct - ref_pct) * float(math.log(recent_pct / ref_pct))
    return total

# COMMAND ----------

features = spark.table(features_ref).filter(F.col("open_time").isNotNull())
latest_time = features.agg(F.max("open_time").alias("latest_time")).collect()[0]["latest_time"]
total_rows = features.count()

print(f"latest_time={latest_time}")
print(f"total_rows={total_rows}")

drift_features = ["volume", "quote_volume", "trades", "return_1h"]

# COMMAND ----------

results = []

for wc in window_configs:
    recent_hours = wc["recent"]
    reference_hours = wc["reference"]
    label = wc["label"]

    recent_start = latest_time - timedelta(hours=recent_hours)
    reference_start = latest_time - timedelta(hours=recent_hours + reference_hours)

    recent = features.filter(F.col("open_time") > F.lit(recent_start))
    reference = features.filter(
        (F.col("open_time") <= F.lit(recent_start))
        & (F.col("open_time") > F.lit(reference_start))
    )

    recent_count = recent.count()
    reference_count = reference.count()

    if recent_count < 10 or reference_count < 10:
        print(f"SKIP {label}: recent={recent_count}, reference={reference_count} (need >= 10)")
        continue

    print(f"\n{'='*60}")
    print(f"Window: {label} (recent={recent_hours}h, reference={reference_hours}h)")
    print(f"Recent rows: {recent_count}, Reference rows: {reference_count}")
    print(f"{'='*60}")

    for col_name in drift_features:
        if col_name not in features.columns:
            continue

        non_null_reference = reference.filter(F.col(col_name).isNotNull())
        non_null_recent = recent.filter(F.col(col_name).isNotNull())

        if non_null_reference.count() < 10 or non_null_recent.count() < 10:
            print(f"  SKIP {col_name}: insufficient non-null rows")
            continue

        psi_value = psi(non_null_reference, non_null_recent, col_name)
        ks_value = approx_ks(non_null_reference, non_null_recent, col_name)

        print(f"  {col_name}: PSI={psi_value:.4f}, KS={ks_value:.4f}")

        for psi_t in psi_thresholds:
            psi_status = "alert" if psi_value >= psi_t else "ok"
            results.append({
                "window": label,
                "feature": col_name,
                "metric": "PSI",
                "value": round(psi_value, 4),
                "threshold": psi_t,
                "status": psi_status,
            })

        for ks_t in ks_thresholds:
            ks_status = "alert" if ks_value >= ks_t else "ok"
            results.append({
                "window": label,
                "feature": col_name,
                "metric": "KS",
                "value": round(ks_value, 4),
                "threshold": ks_t,
                "status": ks_status,
            })

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, DoubleType

schema = StructType([
    StructField("window", StringType(), True),
    StructField("feature", StringType(), True),
    StructField("metric", StringType(), True),
    StructField("value", DoubleType(), True),
    StructField("threshold", DoubleType(), True),
    StructField("status", StringType(), True),
])

results_df = spark.createDataFrame(results, schema=schema)

# COMMAND ----------

# --- Summary: actual PSI/KS values ---

print("\n=== ACTUAL DRIFT VALUES ===")
display(
    results_df.filter(F.col("metric") == "PSI")
    .groupBy("window", "feature", "metric")
    .agg(F.first("value").alias("psi_value"))
    .orderBy("window", "feature")
)

print("\n=== ACTUAL KS VALUES ===")
display(
    results_df.filter(F.col("metric") == "KS")
    .groupBy("window", "feature", "metric")
    .agg(F.first("value").alias("ks_value"))
    .orderBy("window", "feature")
)

# COMMAND ----------

# --- Alert frequency by threshold ---

print("\n=== PSI ALERT FREQUENCY BY THRESHOLD ===")
psi_summary = results_df.filter(F.col("metric") == "PSI").groupBy("threshold").agg(
    F.count("*").alias("total_checks"),
    F.sum(F.when(F.col("status") == "alert", 1).otherwise(0)).alias("alert_count"),
).withColumn("alert_pct", F.round(F.col("alert_count") / F.col("total_checks") * 100, 1))

display(psi_summary.orderBy("threshold"))

print("\n=== KS ALERT FREQUENCY BY THRESHOLD ===")
ks_summary = results_df.filter(F.col("metric") == "KS").groupBy("threshold").agg(
    F.count("*").alias("total_checks"),
    F.sum(F.when(F.col("status") == "alert", 1).otherwise(0)).alias("alert_count"),
).withColumn("alert_pct", F.round(F.col("alert_count") / F.col("total_checks") * 100, 1))

display(ks_summary.orderBy("threshold"))

# COMMAND ----------

# --- Recommended thresholds ---

print("\n=== RECOMMENDED THRESHOLDS ===")
print("Target: ~10-30% alert rate (not too noisy, not too silent)")
print("")

for metric_name in ["PSI", "KS"]:
    metric_results = results_df.filter(F.col("metric") == metric_name)
    summary = metric_results.groupBy("threshold").agg(
        F.sum(F.when(F.col("status") == "alert", 1).otherwise(0)).alias("alert_count"),
        F.count("*").alias("total"),
    ).withColumn("alert_pct", F.round(F.col("alert_count") / F.col("total") * 100, 1))

    best = summary.filter(
        (F.col("alert_pct") >= 10) & (F.col("alert_pct") <= 30)
    ).orderBy(F.abs(F.col("alert_pct") - 20))

    if best.count() > 0:
        row = best.first()
        print(f"{metric_name}: threshold={row['threshold']} (alert rate={row['alert_pct']}%)")
    else:
        row = summary.orderBy(F.abs(F.col("alert_pct") - 20)).first()
        print(f"{metric_name}: threshold={row['threshold']} (alert rate={row['alert_pct']}%) — closest to 20%")
