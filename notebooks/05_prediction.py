# Databricks notebook source
# MAGIC %md # 05 - Prediction

import mlflow
import pandas as pd
from datetime import datetime
from src.utils.config import load_config

config = load_config()

champion = mlflow.pyfunc.load_model(f"models:/btc_model@Champion")
latest_features = spark.table(
    f"{config['catalog']}.{config['features_schema']}.btc_features"
).tail(1)

prediction = champion.predict(pd.DataFrame([latest_features]))[0]
print(f"Next hour BTC price prediction: ${prediction:.2f}")

pred_df = spark.createDataFrame(
    [{"predicted_close": float(prediction), "prediction_time": datetime.now()}]
)
pred_df.write.mode("append").saveAsTable(
    f"{config['catalog']}.{config['predictions_schema']}.btc_predictions"
)
