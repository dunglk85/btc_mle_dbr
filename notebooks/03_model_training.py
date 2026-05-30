# Databricks notebook source
# MAGIC %md # 03 - Model Training with Optuna

import pandas as pd
from sklearn.model_selection import train_test_split
from src.utils.config import load_config
from src.models.training import optimize

config = load_config()
df = spark.table(
    f"{config['catalog']}.{config['features_schema']}.btc_features"
).toPandas()

target = "close"
feature_cols = [c for c in df.columns if c not in ("open_time", "close_time", target)]
df = df.dropna()

X = df[feature_cols]
y = df[target]

split_idx = int(len(df) * 0.8)
X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

study = optimize(
    X_train, y_train, X_val, y_val,
    model_type=config["model_type"],
    n_trials=config["optuna_n_trials"],
)
print(f"Best RMSE: {study.best_value:.4f}")
print(f"Best params: {study.best_params}")
