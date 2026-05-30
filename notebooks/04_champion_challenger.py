# Databricks notebook source
# MAGIC %md # 04 - Champion vs Challenger

import mlflow
import pandas as pd
from src.utils.config import load_config
from src.models.evaluation import promote_if_better

config = load_config()
df = spark.table(
    f"{config['catalog']}.{config['features_schema']}.btc_features"
).toPandas().dropna()

feature_cols = [c for c in df.columns if c not in ("open_time", "close_time", "close")]
X = df[feature_cols]
y = df["close"]

split_idx = int(len(df) * 0.9)
X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

champion_model = mlflow.search_registered_models(
    filter_string="name='btc_model'"
)
champion_uri = None
if champion_model:
    latest = mlflow.MlflowClient().get_model_version_by_alias("btc_model", "Champion")
    if latest:
        champion_uri = latest.source

challenger_uri = mlflow.search_runs(
    experiment_names=["btc_model_tuning"],
    order_by=["metrics.rmse ASC"],
    max_results=1,
).iloc[0]["artifact_uri"]

challenger = mlflow.pyfunc.load_model(f"{challenger_uri}/model")
promoted, metrics = promote_if_better(challenger, champion_uri, X_test, y_test)

if promoted:
    mlflow.register_model(f"{challenger_uri}/model", "btc_model")
    client = mlflow.MlflowClient()
    client.set_registered_model_alias("btc_model", "Champion", "1")
    print("Challenger promoted to Champion!")
else:
    print("Champion retained.")
