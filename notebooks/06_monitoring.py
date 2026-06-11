# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 06 - Monitoring (Pipeline + Drift)

# COMMAND ----------

import json
from datetime import datetime, timedelta, timezone

from pyspark.sql import Window, functions as F

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_simply")
recent_hours = int(get_widget("recent_hours", "168"))
reference_hours = int(get_widget("reference_hours", "720"))
fail_on_alert = get_widget("fail_on_alert", "false").lower() == "true"
trigger_training_on_drift = get_widget("trigger_training_on_drift", "false").lower() == "true"
training_job_name = get_widget("training_job_name", "")
min_drift_alerts_to_trigger = int(get_widget("min_drift_alerts_to_trigger", "2"))
expected_feature_lookback_loss = int(get_widget("expected_feature_lookback_loss", "168"))

raw_ref = f"{catalog}.raw.btc_hourly"
features_ref = f"{catalog}.features.btc_features"
predictions_ref = f"{catalog}.predictions.btc_predictions"
metrics_ref = f"{catalog}.monitoring.pipeline_metrics"
freshness_threshold_hours = 3

psi_warn_threshold = 0.25
psi_alert_threshold = 1.0
ks_warn_threshold = 0.30
ks_alert_threshold = 0.60
psi_monitor_only_alert_threshold = 999.0
ks_monitor_only_alert_threshold = 999.0
mape_warn_threshold = 0.02
mape_alert_threshold = 0.05
direction_warn_threshold = 0.45
direction_alert_threshold = 0.40

print("RUNNING MONITORING NOTEBOOK")
print(f"raw_ref={raw_ref}")
print(f"features_ref={features_ref}")
print(f"predictions_ref={predictions_ref}")
print(f"metrics_ref={metrics_ref}")
print(f"recent_hours={recent_hours}")
print(f"reference_hours={reference_hours}")
print(f"fail_on_alert={fail_on_alert}")
print(f"trigger_training_on_drift={trigger_training_on_drift}")
print(f"training_job_name={training_job_name}")
print(f"min_drift_alerts_to_trigger={min_drift_alerts_to_trigger}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.monitoring")
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {metrics_ref} (
        metric_time TIMESTAMP,
        metric_name STRING,
        metric_value DOUBLE,
        status STRING,
        details STRING
    )
    USING DELTA
""")

# COMMAND ----------

def table_exists(table_ref):
    try:
        spark.table(table_ref).limit(1).collect()
        return True
    except Exception:
        return False


def latest_table_version(table_ref):
    try:
        row = spark.sql(f"DESCRIBE HISTORY {table_ref} LIMIT 1").collect()[0]
        return {"table": table_ref, "version": int(row["version"]), "timestamp": str(row["timestamp"])}
    except Exception as exc:
        return {"table": table_ref, "version": None, "error": str(exc)}


metric_time = datetime.now(timezone.utc)
lineage_context = {
    "raw": latest_table_version(raw_ref),
    "features": latest_table_version(features_ref),
    "predictions": latest_table_version(predictions_ref) if table_exists(predictions_ref) else None,
}
metrics = []


def append_metric(name, value, status="ok", details=""):
    details_payload = {"message": details, "lineage": lineage_context}
    metrics.append(
        {
            "metric_time": metric_time,
            "metric_name": name,
            "metric_value": float(value) if value is not None else None,
            "status": status,
            "details": json.dumps(details_payload),
        }
    )


def status_by_threshold(value, warn_threshold, alert_threshold, higher_is_worse=True):
    if value is None:
        return "warn"
    if higher_is_worse:
        if value >= alert_threshold:
            return "alert"
        if value >= warn_threshold:
            return "warn"
    else:
        if value <= alert_threshold:
            return "alert"
        if value <= warn_threshold:
            return "warn"
    return "ok"


def approx_ks(reference_df, recent_df, col_name):
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
        ref_bucket = reference_df
        recent_bucket = recent_df
        if lower is not None:
            ref_bucket = ref_bucket.filter(F.col(col_name) > lower)
            recent_bucket = recent_bucket.filter(F.col(col_name) > lower)
        if upper is not None:
            ref_bucket = ref_bucket.filter(F.col(col_name) <= upper)
            recent_bucket = recent_bucket.filter(F.col(col_name) <= upper)
        ref_pct = max(ref_bucket.count() / ref_count, epsilon)
        recent_pct = max(recent_bucket.count() / recent_count, epsilon)
        total += (recent_pct - ref_pct) * float(__import__("math").log(recent_pct / ref_pct))
    return total

# COMMAND ----------

# --- Pipeline Metrics ---

raw = spark.table(raw_ref)
raw_count = raw.count()
latest_raw = raw.agg(F.max("open_time").alias("latest_raw")).collect()[0]["latest_raw"]
duplicate_raw_count = raw.groupBy("open_time").count().filter(F.col("count") > 1).count()
null_raw_open_time_count = raw.filter(F.col("open_time").isNull()).count()

append_metric("raw_count", raw_count)
append_metric(
    "raw_duplicate_open_time_count",
    duplicate_raw_count,
    "ok" if duplicate_raw_count == 0 else "alert",
)
append_metric(
    "raw_null_open_time_count",
    null_raw_open_time_count,
    "ok" if null_raw_open_time_count == 0 else "alert",
)

if latest_raw is None:
    append_metric("raw_freshness_hours", None, "alert", "No raw data")
else:
    latest_raw_utc = latest_raw.replace(tzinfo=timezone.utc)
    freshness_hours = (datetime.now(timezone.utc) - latest_raw_utc).total_seconds() / 3600
    append_metric(
        "raw_freshness_hours",
        freshness_hours,
        "ok" if freshness_hours <= freshness_threshold_hours else "alert",
        f"latest_raw={latest_raw}",
    )

# COMMAND ----------

features = spark.table(features_ref)
features_count = features.count()
features_target_null_count = features.filter(F.col("target_close_1h").isNull()).count()

append_metric("features_count", features_count)
append_metric(
    "features_target_close_1h_null_count",
    features_target_null_count,
    "ok" if features_target_null_count <= 1 else "alert",
    "Expected only rows without exact next-hour candles to have null target.",
)
append_metric(
    "raw_features_row_count_delta",
    raw_count - features_count,
    "ok" if 0 <= (raw_count - features_count) <= expected_feature_lookback_loss + 1 else "alert",
    f"Expected feature lookback loss <= {expected_feature_lookback_loss + 1} rows.",
)

# COMMAND ----------

if table_exists(predictions_ref):
    predictions = spark.table(predictions_ref)
    prediction_count = predictions.count()
    latest_prediction_time = predictions.agg(
        F.max("prediction_time").alias("latest_prediction_time")
    ).collect()[0]["latest_prediction_time"]
    append_metric("prediction_count", prediction_count)
    if latest_prediction_time is not None:
        latest_prediction_utc = latest_prediction_time.replace(tzinfo=timezone.utc)
        prediction_age_hours = (
            datetime.now(timezone.utc) - latest_prediction_utc
        ).total_seconds() / 3600
        append_metric(
            "prediction_age_hours",
            prediction_age_hours,
            "ok" if prediction_age_hours <= freshness_threshold_hours else "alert",
            f"latest_prediction_time={latest_prediction_time}",
        )
    else:
        append_metric("prediction_age_hours", None, "alert", "No predictions")
else:
    append_metric("prediction_count", 0, "warn", "Predictions table does not exist")

# COMMAND ----------

# --- Drift Metrics ---

if not table_exists(features_ref):
    append_metric("drift_features_table_exists", 0, "alert", f"Missing {features_ref}")
else:
    features = spark.table(features_ref).filter(F.col("open_time").isNotNull())
    latest_time = features.agg(F.max("open_time").alias("latest_time")).collect()[0]["latest_time"]
    if latest_time is None:
        append_metric("drift_features_row_count", 0, "alert", "No feature rows")
    else:
        recent_start = latest_time - timedelta(hours=recent_hours)
        reference_start = latest_time - timedelta(hours=recent_hours + reference_hours)
        recent = features.filter(F.col("open_time") > F.lit(recent_start))
        reference = features.filter(
            (F.col("open_time") <= F.lit(recent_start))
            & (F.col("open_time") > F.lit(reference_start))
        )

        recent_count = recent.count()
        reference_count = reference.count()
        append_metric("drift_recent_feature_count", recent_count, "ok")
        append_metric("drift_reference_feature_count", reference_count, "ok")

        drift_features = [
            "volume",
            "quote_volume",
            "trades",
            "return_1h",
        ]
        for col_name in drift_features:
            if col_name not in features.columns:
                append_metric(f"schema_drift_missing_{col_name}", 1, "alert", "Missing feature")
                continue
            non_null_reference = reference.filter(F.col(col_name).isNotNull())
            non_null_recent = recent.filter(F.col(col_name).isNotNull())
            null_rate = 1.0
            if recent_count > 0:
                null_rate = recent.filter(F.col(col_name).isNull()).count() / recent_count
            append_metric(
                f"feature_quality_null_rate_{col_name}",
                null_rate,
                status_by_threshold(null_rate, 0.05, 0.2),
                "Recent-window null rate",
            )
            if non_null_reference.count() == 0 or non_null_recent.count() == 0:
                append_metric(f"data_drift_psi_{col_name}", None, "warn", "Insufficient rows")
                append_metric(f"data_drift_ks_{col_name}", None, "warn", "Insufficient rows")
                continue
            psi_value = psi(non_null_reference, non_null_recent, col_name)
            ks_value = approx_ks(non_null_reference, non_null_recent, col_name)
            append_metric(
                f"data_drift_psi_{col_name}",
                psi_value,
                status_by_threshold(psi_value, psi_warn_threshold, psi_alert_threshold),
                f"reference_hours={reference_hours}; recent_hours={recent_hours}",
            )
            append_metric(
                f"data_drift_ks_{col_name}",
                ks_value,
                status_by_threshold(ks_value, ks_warn_threshold, ks_alert_threshold),
                f"reference_hours={reference_hours}; recent_hours={recent_hours}",
            )

        monitor_only_features = ["close", "ma_24", "target_close_1h"]
        for col_name in monitor_only_features:
            if col_name not in features.columns:
                append_metric(f"schema_drift_missing_{col_name}", 1, "alert", "Missing feature")
                continue
            non_null_reference = reference.filter(F.col(col_name).isNotNull())
            non_null_recent = recent.filter(F.col(col_name).isNotNull())
            if non_null_reference.count() == 0 or non_null_recent.count() == 0:
                continue
            psi_value = psi(non_null_reference, non_null_recent, col_name)
            ks_value = approx_ks(non_null_reference, non_null_recent, col_name)
            metric_prefix = "label_drift" if col_name == "target_close_1h" else "price_level_drift"
            append_metric(
                f"{metric_prefix}_psi_{col_name}",
                psi_value,
                status_by_threshold(psi_value, psi_warn_threshold, psi_monitor_only_alert_threshold),
                "Monitor-only price/label level drift; not a retrain trigger",
            )
            append_metric(
                f"{metric_prefix}_ks_{col_name}",
                ks_value,
                status_by_threshold(ks_value, ks_warn_threshold, ks_monitor_only_alert_threshold),
                "Monitor-only price/label level drift; not a retrain trigger",
            )

# COMMAND ----------

if table_exists(predictions_ref) and table_exists(raw_ref):
    predictions = spark.table(predictions_ref)
    raw = spark.table(raw_ref)
    joined = predictions.alias("p").join(
        raw.alias("r"),
        F.col("r.open_time") == F.col("p.feature_open_time") + F.expr("INTERVAL 1 HOUR"),
        "inner",
    )
    if joined.count() == 0:
        append_metric("model_drift_joined_prediction_count", 0, "warn", "No actuals yet")
    else:
        latest_prediction = joined.agg(
            F.max("p.prediction_time").alias("latest_prediction")
        ).collect()[0]["latest_prediction"]
        recent_predictions = joined
        if latest_prediction is not None:
            prediction_start = latest_prediction - timedelta(hours=recent_hours)
            recent_predictions = joined.filter(F.col("p.prediction_time") > F.lit(prediction_start))

        latest_prediction_time = predictions.agg(
            F.max("prediction_time").alias("latest_prediction_time")
        ).collect()[0]["latest_prediction_time"]
        if latest_prediction_time is not None:
            prediction_recent_start = latest_prediction_time - timedelta(hours=recent_hours)
            prediction_reference_start = latest_prediction_time - timedelta(
                hours=recent_hours + reference_hours
            )
            recent_prediction_dist = predictions.filter(
                F.col("prediction_time") > F.lit(prediction_recent_start)
            ).filter(F.col("predicted_close").isNotNull())
            reference_prediction_dist = predictions.filter(
                (F.col("prediction_time") <= F.lit(prediction_recent_start))
                & (F.col("prediction_time") > F.lit(prediction_reference_start))
            ).filter(F.col("predicted_close").isNotNull())
            prediction_psi = psi(
                reference_prediction_dist, recent_prediction_dist, "predicted_close"
            )
            prediction_ks = approx_ks(
                reference_prediction_dist, recent_prediction_dist, "predicted_close"
            )
            append_metric(
                "prediction_drift_psi_predicted_close",
                prediction_psi,
                status_by_threshold(prediction_psi, psi_warn_threshold, psi_monitor_only_alert_threshold),
                "Monitor-only prediction level drift; not a retrain trigger",
            )
            append_metric(
                "prediction_drift_ks_predicted_close",
                prediction_ks,
                status_by_threshold(prediction_ks, ks_warn_threshold, ks_monitor_only_alert_threshold),
                "Monitor-only prediction level drift; not a retrain trigger",
            )

        scored = recent_predictions.select(
            F.col("p.predicted_close").alias("predicted_close"),
            F.col("r.close").alias("actual_close"),
            F.col("p.feature_open_time").alias("feature_open_time"),
        )
        invalid_close_predictions = scored.filter(F.col("predicted_close") <= 1000.0).count()
        if invalid_close_predictions:
            append_metric(
                f"model_drift_invalid_predicted_close_count_{recent_hours}h",
                invalid_close_predictions,
                "warn",
                "Ignored predictions that are too small to be BTC close prices; likely legacy return-as-close rows",
            )
        scored = scored.filter(F.col("predicted_close") > 1000.0).withColumn(
            "error",
            F.col("actual_close") - F.col("predicted_close"),
        )

        scored = scored.withColumn("abs_error", F.abs(F.col("error"))).withColumn(
            "pct_error", F.abs(F.col("error")) / F.abs(F.col("actual_close"))
        )
        perf = scored.agg(
            F.count("*").alias("count"),
            F.sqrt(F.avg(F.pow(F.col("error"), 2))).alias("rmse"),
            F.avg("abs_error").alias("mae"),
            F.avg("pct_error").alias("mape"),
            F.avg("error").alias("mean_error"),
            F.percentile_approx("abs_error", 0.95).alias("p95_abs_error"),
        ).collect()[0]
        actual_mean = scored.agg(F.avg("actual_close").alias("actual_mean")).collect()[0]["actual_mean"]
        r2 = None
        if actual_mean is not None:
            r2_parts = scored.agg(
                F.sum(F.pow(F.col("actual_close") - F.col("predicted_close"), 2)).alias("ss_res"),
                F.sum(F.pow(F.col("actual_close") - F.lit(actual_mean), 2)).alias("ss_tot"),
            ).collect()[0]
            if r2_parts["ss_tot"] and float(r2_parts["ss_tot"]) != 0.0:
                r2 = 1.0 - (float(r2_parts["ss_res"]) / float(r2_parts["ss_tot"]))
        append_metric("model_drift_joined_prediction_count", perf["count"], "ok")
        append_metric(f"model_drift_rmse_{recent_hours}h", perf["rmse"], "ok")
        append_metric(f"model_drift_mae_{recent_hours}h", perf["mae"], "ok")
        append_metric(f"model_drift_r2_{recent_hours}h", r2, "ok")
        append_metric(
            f"model_drift_mape_{recent_hours}h",
            perf["mape"],
            status_by_threshold(perf["mape"], mape_warn_threshold, mape_alert_threshold),
        )
        append_metric(f"model_drift_p95_abs_error_{recent_hours}h", perf["p95_abs_error"], "ok")
        append_metric(
            f"concept_drift_mean_error_bias_{recent_hours}h",
            perf["mean_error"],
            "ok",
            "Proxy metric: persistent signed error can indicate concept drift",
        )

        direction_window = Window.orderBy("feature_open_time")
        direction = scored.withColumn(
            "actual_direction",
            F.signum(
                F.col("actual_close")
                - F.lag("actual_close").over(direction_window)
            ),
        ).withColumn(
            "predicted_direction",
            F.signum(
                F.col("predicted_close")
                - F.lag("predicted_close").over(direction_window)
            ),
        )
        direction_accuracy = direction.filter(
            F.col("actual_direction").isNotNull() & F.col("predicted_direction").isNotNull()
        ).agg(
            F.avg((F.col("actual_direction") == F.col("predicted_direction")).cast("double")).alias("accuracy")
        ).collect()[0]["accuracy"]
        append_metric(
            f"model_drift_direction_accuracy_{recent_hours}h",
            direction_accuracy,
            status_by_threshold(direction_accuracy, direction_warn_threshold, direction_alert_threshold, higher_is_worse=False),
        )
else:
    append_metric("model_drift_joined_prediction_count", 0, "warn", "Predictions or raw table missing")

# COMMAND ----------

# --- Drift-Triggered Training ---

pre_trigger_metrics_df = spark.createDataFrame(metrics)
pre_trigger_alert_count = pre_trigger_metrics_df.filter(F.col("status") == "alert").count()

if not trigger_training_on_drift:
    append_metric("training_trigger_status", 0, "ok", "trigger_training_on_drift=false")
    print("trigger_training_on_drift=false; skipping training trigger check")
else:
    drift_alert_count = pre_trigger_metrics_df.filter(
        F.col("status") == "alert"
    ).filter(
        F.col("metric_name").rlike("^(data_drift|label_drift|prediction_drift|model_drift|concept_drift)_")
    ).count()

    print(f"drift_alert_count={drift_alert_count}")
    print(f"min_drift_alerts_to_trigger={min_drift_alerts_to_trigger}")

    append_metric(
        "training_trigger_drift_alert_count",
        drift_alert_count,
        "ok" if drift_alert_count < min_drift_alerts_to_trigger else "warn",
        f"min_drift_alerts_to_trigger={min_drift_alerts_to_trigger}",
    )

    if drift_alert_count >= min_drift_alerts_to_trigger:
        print(f"DRIFT_THRESHOLD_MET: {drift_alert_count} drift alerts >= {min_drift_alerts_to_trigger}")

        if training_job_name:
            try:
                try:
                    from databricks.sdk import WorkspaceClient
                except ImportError as exc:
                    raise RuntimeError("databricks-sdk is required to trigger training job") from exc

                workspace = WorkspaceClient()
                matches = list(workspace.jobs.list(name=training_job_name))
                if not matches:
                    raise ValueError(f"Could not find Databricks job named '{training_job_name}'")
                if len(matches) > 1:
                    raise ValueError(f"Found multiple Databricks jobs named '{training_job_name}'")

                resolved_job_id = matches[0].job_id
                print(f"resolved_job_name='{training_job_name}' → job_id={resolved_job_id}")

                run = workspace.jobs.run_now(job_id=resolved_job_id)
                print(f"training_job_triggered=true run_id={run.run_id}")
                append_metric(
                    "training_trigger_status",
                    1,
                    "ok",
                    f"Triggered training_job_name='{training_job_name}' (id={resolved_job_id}); run_id={run.run_id}",
                )
            except Exception as exc:
                print(f"training_job_trigger_failed={exc}")
                append_metric(
                    "training_trigger_status",
                    0,
                    "warn",
                    f"Failed to trigger training_job_name='{training_job_name}': {exc}",
                )
        else:
            message = "training_job_name not set; training must be triggered manually or via scheduled run"
            print(message)
            append_metric("training_trigger_status", 0, "warn", message)
    else:
        append_metric("training_trigger_status", 0, "ok", "DRIFT_THRESHOLD_NOT_MET")
        print("DRIFT_THRESHOLD_NOT_MET; no training trigger needed")

# COMMAND ----------

metrics_df = spark.createDataFrame(metrics)
metrics_df.write.mode("append").saveAsTable(metrics_ref)

alert_count = metrics_df.filter(F.col("status") == "alert").count()
warn_count = metrics_df.filter(F.col("status") == "warn").count()

print(f"metrics_written={metrics_df.count()}")
print(f"alert_count={alert_count}")
print(f"warn_count={warn_count}")

display(metrics_df.orderBy("metric_name"))

if fail_on_alert and alert_count > 0:
    raise ValueError(f"Monitoring produced {alert_count} alert metrics")

if alert_count > 0:
    print(f"ALERTS_RECORDED: {alert_count} alert metrics written; fail_on_alert=false so job continues")
