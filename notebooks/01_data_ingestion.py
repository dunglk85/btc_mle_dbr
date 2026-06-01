# Databricks notebook source
# MAGIC %md # 01 - Data Ingestion

from src.utils.config import load_config
from src.data.ingestion import load_landing_to_raw

spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

config = load_config()
df = load_landing_to_raw(
    spark,
    catalog=config["catalog"],
    raw_schema=config["raw_schema"],
    volume_name=config["landing_volume"],
    table="btc_hourly",
)
display(df.tail(10))
