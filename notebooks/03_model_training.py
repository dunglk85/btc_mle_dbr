# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 - Baseline Model Training

# COMMAND ----------

import mlflow
from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# COMMAND ----------

catalog = "btc_dev"
features_schema = "features"
features_table = "btc_features"
features_ref = f"{catalog}.{features_schema}.{features_table}"

print("RUNNING SELF-CONTAINED BASELINE TRAINING NOTEBOOK")
print(f"features_ref={features_ref}")

# COMMAND ----------

source = spark.table(features_ref).orderBy("open_time")
feature_cols = [
    "open",
    "high",
    "low",
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

model_df = source.select("open_time", "close", *feature_cols).dropna()
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
y_train = train["close"]
X_test = test[feature_cols]
y_test = test["close"]

# COMMAND ----------

model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

mlflow.set_experiment("/Shared/btc_baseline_training")
with mlflow.start_run(run_name="baseline_random_forest") as run:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    mape = float((abs(y_test - preds) / y_test.abs()).mean())

    mlflow.log_param("model_type", "random_forest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("mape", mape)
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature)
    run_id = run.info.run_id

# COMMAND ----------

print(f"run_id={run_id}")
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
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
            }
        ]
    )
)
