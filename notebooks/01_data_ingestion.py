# Databricks notebook source
# MAGIC %md # 01 - Data Ingestion

import sys

notebook_path = (
    dbutils.notebook.entry_point.getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)
repo_root = "/Workspace" + "/".join(notebook_path.split("/")[:-2])
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

print(f"RUNNING INGESTION NOTEBOOK: {notebook_path}")
print(f"repo_root={repo_root}")

from src.utils.config import load_config
from src.data.ingestion import load_landing_to_raw

spark.conf.set("spark.sql.sources.partitionOverwriteMode", "dynamic")

config = load_config()
landing_path = f"/Volumes/{config['catalog']}/{config['raw_schema']}/{config['landing_volume']}/btc_hourly"
landing_count = spark.read.option("header", True).csv(landing_path).count()
print(f"landing_path={landing_path}")
print(f"landing_count_before_merge={landing_count}")

df = load_landing_to_raw(
    spark,
    catalog=config["catalog"],
    raw_schema=config["raw_schema"],
    volume_name=config["landing_volume"],
    table="btc_hourly",
)
print(f"raw_table_count_after_merge={df.count()}")
display(df.orderBy("open_time", ascending=False).limit(10))
