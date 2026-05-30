# Databricks notebook source
# MAGIC %md # 01 - Data Ingestion

from src.utils.config import load_config
from src.data.ingestion import incremental_ingest

spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

config = load_config()
df = incremental_ingest(
    spark,
    catalog=config["catalog"],
    schema=config["raw_schema"],
    table="btc_hourly",
)
display(df.tail(10))
