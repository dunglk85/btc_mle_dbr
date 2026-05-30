# Databricks notebook source
# MAGIC %md # 02 - Feature Engineering

from src.utils.config import load_config
from src.data.features import compute_features

config = load_config()
raw_df = spark.table(f"{config['catalog']}.{config['raw_schema']}.btc_hourly")
features_df = compute_features(raw_df)
features_df.write.mode("overwrite").saveAsTable(
    f"{config['catalog']}.{config['features_schema']}.btc_features"
)
display(features_df.tail(10))
