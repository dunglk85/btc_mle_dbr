# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 - Optuna Model Training

# COMMAND ----------

# MAGIC %pip install optuna

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
import optuna
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# COMMAND ----------

def get_widget(name, default):
    try:
        dbutils.widgets.text(name, str(default))
        return dbutils.widgets.get(name)
    except Exception:
        return str(default)


catalog = get_widget("catalog", "btc_dev")
features_schema = "features"
features_table = "btc_features"
features_ref = f"{catalog}.{features_schema}.{features_table}"
decisions_ref = f"{catalog}.monitoring.model_refresh_decisions"
experiment_name = "/Shared/btc_baseline_training"
default_n_trials = 15
default_timeout_seconds = 900


n_trials = int(get_widget("n_trials", default_n_trials))
timeout_seconds = int(get_widget("timeout_seconds", default_timeout_seconds))
if n_trials < 1:
    raise ValueError(f"n_trials must be >= 1, got {n_trials}")
if timeout_seconds <= 0:
    raise ValueError(f"timeout_seconds must be > 0, got {timeout_seconds}")

print("RUNNING SELF-CONTAINED OPTUNA TRAINING NOTEBOOK")
print(f"features_ref={features_ref}")
print(f"n_trials={n_trials}")
print(f"timeout_seconds={timeout_seconds}")

# COMMAND ----------

skip_reason = None
try:
    latest_decision = spark.table(decisions_ref).orderBy(
        "decision_time", ascending=False
    ).limit(1).collect()
    if latest_decision and not latest_decision[0]["should_retrain"]:
        skip_reason = latest_decision[0]["reason"]
except Exception as exc:
    print(f"No monitoring gate decision found, continuing training: {exc}")

if skip_reason:
    print(f"SKIP_RETRAIN: {skip_reason}")
    dbutils.jobs.taskValues.set(key="training_status", value="skipped")
    dbutils.notebook.exit("SKIP_RETRAIN")

# COMMAND ----------

source = spark.table(features_ref).orderBy("open_time")
target_col = "target_close_1h"
feature_cols = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "ma_7",
    "ma_24",
    "ma_168",
    "close_lag_1h",
    "close_lag_2h",
    "close_lag_4h",
    "close_lag_12h",
    "close_lag_24h",
    "return_1h",
    "hl_spread",
    "oc_change",
    "hour",
    "day_of_week",
]

model_df = source.select("open_time", target_col, *feature_cols).dropna()
row_count = model_df.count()
print(f"training_rows_after_dropna={row_count}")
if row_count < 100:
    raise ValueError(f"Not enough training rows in {features_ref}: {row_count}")

# COMMAND ----------

pdf = model_df.toPandas().sort_values("open_time")
split_idx = int(len(pdf) * 0.8)
train = pdf.iloc[:split_idx]
test = pdf.iloc[split_idx:]

X_train = train[feature_cols]
y_train = train[target_col]
X_test = test[feature_cols]
y_test = test[target_col]

# COMMAND ----------

mlflow.set_experiment(experiment_name)


def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
    }

    with mlflow.start_run(run_name="optuna_random_forest_trial", nested=True):
        mlflow.log_param("model_type", "random_forest")
        mlflow.log_params(params)
        model = RandomForestRegressor(**params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae = mean_absolute_error(y_test, preds)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
    return rmse


with mlflow.start_run(run_name="optuna_random_forest_parent") as parent_run:
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout_seconds)

    best_params = study.best_params
    best_model = RandomForestRegressor(**best_params, random_state=42)
    best_model.fit(X_train, y_train)
    best_preds = best_model.predict(X_test)
    rmse = mean_squared_error(y_test, best_preds) ** 0.5
    mae = mean_absolute_error(y_test, best_preds)
    r2 = r2_score(y_test, best_preds)
    mape = float((abs(y_test - best_preds) / y_test.abs()).mean())

    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("training_mode", "optuna")
    mlflow.log_param("n_trials_requested", n_trials)
    mlflow.log_param("n_trials_completed", len(study.trials))
    mlflow.log_params(best_params)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)
    signature = infer_signature(X_train, best_model.predict(X_train))
    mlflow.sklearn.log_model(best_model, "model", signature=signature)
    run_id = parent_run.info.run_id

dbutils.jobs.taskValues.set(key="training_status", value="trained")
dbutils.jobs.taskValues.set(key="run_id", value=run_id)

# COMMAND ----------

print(f"run_id={run_id}")
print(f"best_params={best_params}")
print(f"n_trials_completed={len(study.trials)}")
print(f"rmse={rmse:.4f}")
print(f"mae={mae:.4f}")
print(f"r2={r2:.4f}")
print(f"mape={mape:.6f}")

# COMMAND ----------

display(
    spark.createDataFrame(
        [
            {
                "run_id": run_id,
                "rmse": float(rmse),
                "mae": float(mae),
                "r2": float(r2),
                "mape": float(mape),
                "n_trials_requested": int(n_trials),
                "n_trials_completed": int(len(study.trials)),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
            }
        ]
    )
)
