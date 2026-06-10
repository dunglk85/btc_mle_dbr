# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 12 - Training Dataset Replay Validation

# COMMAND ----------

import json
import re
from datetime import timezone

from pyspark.sql import Window, functions as F

# COMMAND ----------


def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_simply")
training_task_key = get_widget("training_task_key", "model_training")
min_delta_retention_hours = int(get_widget("min_delta_retention_hours", "168"))

training_manifest_ref = f"{catalog}.monitoring.training_dataset_manifests"

print("RUNNING TRAINING DATASET REPLAY VALIDATION")
print(f"training_manifest_ref={training_manifest_ref}")
print(f"training_task_key={training_task_key}")
print(f"min_delta_retention_hours={min_delta_retention_hours}")

# COMMAND ----------

run_id = dbutils.jobs.taskValues.get(
    taskKey=training_task_key,
    key="run_id",
    default="",
)
training_status = dbutils.jobs.taskValues.get(
    taskKey=training_task_key,
    key="training_status",
    default="unknown",
)

if training_status != "trained":
    print(f"SKIP_DATASET_REPLAY: training_status={training_status}")
    dbutils.notebook.exit("SKIP_DATASET_REPLAY")
if not run_id:
    raise ValueError(f"Missing run_id task value from {training_task_key}")

manifest_rows = (
    spark.table(training_manifest_ref)
    .filter(F.col("run_id") == run_id)
    .orderBy(F.col("created_at").desc())
    .limit(1)
    .collect()
)
if not manifest_rows:
    raise ValueError(f"No training dataset manifest found for run_id={run_id}")

manifest = manifest_rows[0].asDict()
feature_cols = json.loads(manifest["feature_cols_json"])
target_col = manifest["target_col"]

print(f"run_id={run_id}")
print(f"raw_table={manifest['raw_table']}@v{manifest['raw_table_version']}")
print(f"features_table={manifest['features_table']}@v{manifest['features_table_version']}")
print(f"feature_config_table={manifest['feature_config_table']}@v{manifest['feature_config_version']}")
print(f"feature_config_id={manifest.get('feature_config_id')}")
print(f"n_features={len(feature_cols)}")

# COMMAND ----------


def assert_delta_version_available(table_ref, version):
    try:
        spark.read.option("versionAsOf", int(version)).table(table_ref).limit(1).collect()
    except Exception as exc:
        raise ValueError(
            f"Delta version unavailable for replay: {table_ref}@v{version}. "
            "Check VACUUM/retention policy before claiming reproducibility."
        ) from exc


def interval_to_hours(value):
    if not value:
        raise ValueError("empty interval")

    normalized = value.strip().strip("'").strip('"').lower()
    if not normalized.startswith("interval"):
        raise ValueError(f"unsupported interval format: {value}")

    body = normalized[len("interval"):].strip()
    token_pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s+"
        r"(week|weeks|day|days|hour|hours|minute|minutes|second|seconds)"
    )
    total_hours = 0.0
    consumed = []
    for match in token_pattern.finditer(body):
        amount = float(match.group(1))
        unit = match.group(2)
        consumed.append(match.group(0))
        if unit.startswith("week"):
            total_hours += amount * 24 * 7
        elif unit.startswith("day"):
            total_hours += amount * 24
        elif unit.startswith("hour"):
            total_hours += amount
        elif unit.startswith("minute"):
            total_hours += amount / 60
        elif unit.startswith("second"):
            total_hours += amount / 3600

    remainder = body
    for token in consumed:
        remainder = remainder.replace(token, "", 1)
    if not consumed or remainder.strip():
        raise ValueError(f"unsupported interval format: {value}")
    return total_hours


def check_retention(table_ref):
    try:
        detail = spark.sql(f"DESCRIBE DETAIL {table_ref}").collect()[0].asDict()
    except Exception as exc:
        raise ValueError(f"Could not read Delta retention properties for {table_ref}: {exc}") from exc

    properties = detail.get("properties") or {}
    deleted_retention = properties.get("delta.deletedFileRetentionDuration", "interval 1 week")
    log_retention = properties.get("delta.logRetentionDuration", "interval 30 days")
    print(f"{table_ref} deletedFileRetention={deleted_retention}; logRetention={log_retention}")

    deleted_retention_hours = interval_to_hours(deleted_retention)
    log_retention_hours = interval_to_hours(log_retention)
    if deleted_retention_hours < min_delta_retention_hours:
        raise ValueError(
            f"Delta deleted file retention too short for replay: {table_ref} has "
            f"{deleted_retention} ({deleted_retention_hours}h), expected >= "
            f"{min_delta_retention_hours}h"
        )
    if log_retention_hours < min_delta_retention_hours:
        raise ValueError(
            f"Delta log retention too short for replay: {table_ref} has "
            f"{log_retention} ({log_retention_hours}h), expected >= "
            f"{min_delta_retention_hours}h"
        )


for table_ref, version in [
    (manifest["raw_table"], manifest["raw_table_version"]),
    (manifest["features_table"], manifest["features_table_version"]),
    (manifest["feature_config_table"], manifest["feature_config_version"]),
]:
    if int(version) < 0:
        print(f"SKIP_OPTIONAL_DELTA_VERSION_CHECK: {table_ref}@v{version}")
        continue
    assert_delta_version_available(table_ref, version)
    check_retention(table_ref)

# COMMAND ----------

features = spark.read.option("versionAsOf", int(manifest["features_table_version"])).table(
    manifest["features_table"]
)
missing_features = [col for col in feature_cols if col not in features.columns]
if missing_features:
    raise ValueError(f"Manifest features missing in replayed feature table: {missing_features}")
if target_col not in features.columns:
    raise ValueError(f"Manifest target column missing in replayed feature table: {target_col}")

replayed = features.select("open_time", target_col, *feature_cols).dropna()
replayed_count = replayed.count()
expected_count = int(manifest["train_rows"]) + int(manifest["test_rows"])
if replayed_count != expected_count:
    raise ValueError(
        f"Replay row count mismatch: replayed={replayed_count}, expected={expected_count}. "
        "This usually means manifest split logic and replay logic diverged."
    )

duplicate_open_time_count = (
    replayed.groupBy("open_time")
    .count()
    .filter(F.col("count") > 1)
    .count()
)
if duplicate_open_time_count > 0:
    raise ValueError(
        f"Replay data contains {duplicate_open_time_count} duplicate open_time values; "
        "train/test boundary ordering is not deterministic."
    )

manifest_split_idx = manifest.get("split_idx")
if manifest_split_idx is not None:
    split_idx = int(manifest_split_idx)
else:
    train_fraction = float(manifest.get("train_fraction") or 0.8)
    split_idx = int(replayed_count * train_fraction)
if split_idx != int(manifest["train_rows"]):
    raise ValueError(f"Replay split_idx mismatch: {split_idx} != train_rows={manifest['train_rows']}")

indexed = replayed.select("open_time").withColumn(
    "_row_number",
    F.row_number().over(Window.orderBy("open_time")),
)
train_bounds = indexed.filter(F.col("_row_number") <= split_idx).agg(
    F.min("open_time").alias("train_start_time"),
    F.max("open_time").alias("train_end_time"),
    F.count("*").alias("train_rows"),
).collect()[0].asDict()
test_bounds = indexed.filter(F.col("_row_number") > split_idx).agg(
    F.min("open_time").alias("test_start_time"),
    F.max("open_time").alias("test_end_time"),
    F.count("*").alias("test_rows"),
).collect()[0].asDict()
if int(train_bounds["train_rows"]) == 0 or int(test_bounds["test_rows"]) == 0:
    raise ValueError(
        f"Replay split produced empty train/test partition: "
        f"train={train_bounds['train_rows']}, test={test_bounds['test_rows']}"
    )

def timestamp_to_epoch_micros(value):
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return int(value.timestamp() * 1_000_000)


def assert_timestamp_match(name, replay_value, manifest_value):
    replay_micros = timestamp_to_epoch_micros(replay_value)
    manifest_micros = timestamp_to_epoch_micros(manifest_value)
    if replay_micros != manifest_micros:
        raise ValueError(f"Replay {name} mismatch: {replay_value} != {manifest_value}")


assert_timestamp_match("train_start_time", train_bounds["train_start_time"], manifest["train_start_time"])
assert_timestamp_match("train_end_time", train_bounds["train_end_time"], manifest["train_end_time"])
assert_timestamp_match("test_start_time", test_bounds["test_start_time"], manifest["test_start_time"])
assert_timestamp_match("test_end_time", test_bounds["test_end_time"], manifest["test_end_time"])

if int(manifest["n_features"]) != len(feature_cols):
    raise ValueError(f"Manifest n_features mismatch: {manifest['n_features']} != {len(feature_cols)}")

print(f"REPLAY_VALIDATED: run_id={run_id}; rows={replayed_count}; features={len(feature_cols)}")
dbutils.jobs.taskValues.set(key="dataset_replay_status", value="validated")
dbutils.jobs.taskValues.set(key="run_id", value=run_id)

display(
    spark.createDataFrame(
        [
            {
                "run_id": run_id,
                "status": "validated",
                "replayed_rows": int(replayed_count),
                "expected_rows": int(expected_count),
                "n_features": int(len(feature_cols)),
                "raw_table_version": int(manifest["raw_table_version"]),
                "features_table_version": int(manifest["features_table_version"]),
                "feature_config_version": int(manifest["feature_config_version"]),
                "feature_config_id": int(manifest.get("feature_config_id") or -1),
            }
        ]
    )
)
