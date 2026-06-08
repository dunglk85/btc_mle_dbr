# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02b - EDA & Feature Selection
# MAGIC
# MAGIC Notebook phân tích khám phá dữ liệu (EDA) và lựa chọn features tối ưu
# MAGIC cho bài toán **Regression** (`target_return_1h`).
# MAGIC
# MAGIC **Phương pháp:**
# MAGIC 1. Kiểm tra phân phối & thống kê cơ bản các features
# MAGIC 2. Correlation Matrix (Pearson & Spearman) — phát hiện collinearity
# MAGIC 3. Mutual Information — đánh giá tương quan phi tuyến với regression target
# MAGIC 4. Feature Importance sơ bộ bằng LightGBM
# MAGIC 5. Xuất danh sách `selected_features` cuối cùng

# COMMAND ----------

# MAGIC %pip install lightgbm scikit-learn matplotlib seaborn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import warnings
warnings.filterwarnings("ignore")

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

# Ngưỡng tương quan cao để loại bỏ collinear features
CORR_THRESHOLD = float(get_widget("corr_threshold", "0.90"))

print("RUNNING EDA & FEATURE SELECTION NOTEBOOK")
print(f"features_ref={features_ref}")
print(f"corr_threshold={CORR_THRESHOLD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load Data & Define Candidate Features

# COMMAND ----------

# Danh sách đầy đủ candidate features
CANDIDATE_FEATURES = [
    # Return features
    "return_1h", "return_6h", "return_24h",
    # Moving Averages & Ratios
    "ma_7", "ma_24", "ma_168",
    "close_ma7_ratio", "close_ma24_ratio", "close_ma168_ratio",
    # MACD
    "macd", "macd_signal", "macd_hist",
    # RSI
    "rsi_14",
    # Volatility
    "atr_14", "atr_ratio", "bb_width",
    # Volume
    "volume_ratio", "log_volume",
    # Spread
    "hl_spread", "oc_change",
    # Lag features
    "close_lag_1h", "close_lag_2h", "close_lag_4h", "close_lag_12h", "close_lag_24h",
    # Time features
    "hour", "day_of_week",
    "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
    # Raw price/volume (sẽ đánh giá xem có nên giữ không)
    "open", "high", "low", "close", "volume", "quote_volume", "trades",
]

REGRESSION_TARGET = "target_return_1h"

# Load data
source = spark.table(features_ref).orderBy("open_time")
features_table_version = int(spark.sql(f"DESCRIBE HISTORY {features_ref} LIMIT 1").collect()[0]["version"])
all_cols = ["open_time", REGRESSION_TARGET] + CANDIDATE_FEATURES
# Chỉ lấy các cột thực sự có trong bảng
existing_cols = [c for c in all_cols if c in source.columns]
missing_cols = [c for c in all_cols if c not in source.columns]
if missing_cols:
    print(f"WARNING: Missing columns (skipped): {missing_cols}")

pdf = source.select(existing_cols).toPandas().sort_values("open_time").reset_index(drop=True)

# Loại bỏ rows có target null (dòng cuối cùng do shift)
pdf = pdf.dropna(subset=[REGRESSION_TARGET])
print(f"Total rows for EDA: {len(pdf)}")
print(f"features_table_version={features_table_version}")

# Cập nhật danh sách features thực tế
feature_cols = [c for c in CANDIDATE_FEATURES if c in pdf.columns]
print(f"Candidate features count: {len(feature_cols)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Basic Statistics & Missing Values

# COMMAND ----------

# Thống kê cơ bản
stats = pdf[feature_cols].describe().T
stats["null_count"] = pdf[feature_cols].isnull().sum()
stats["null_pct"] = (stats["null_count"] / len(pdf) * 100).round(2)
display(spark.createDataFrame(stats.reset_index().rename(columns={"index": "feature"})))

# COMMAND ----------

# Loại bỏ rows có bất kỳ feature nào null (chủ yếu do rolling window đầu)
pdf_clean = pdf[feature_cols + [REGRESSION_TARGET]].dropna()
print(f"Rows after dropping NaN: {len(pdf_clean)} (dropped {len(pdf) - len(pdf_clean)} rows)")

X = pdf_clean[feature_cols]
y_reg = pdf_clean[REGRESSION_TARGET]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Feature Distribution — Phân phối

# COMMAND ----------

# Histogram các feature chính
key_features = [
    "return_1h", "return_24h", "close_ma7_ratio", "close_ma24_ratio",
    "macd", "macd_hist", "rsi_14", "atr_ratio", "bb_width",
    "volume_ratio", "log_volume",
]
key_features = [f for f in key_features if f in feature_cols]

n_cols = 3
n_rows = (len(key_features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
axes = axes.flatten()

for i, feat in enumerate(key_features):
    axes[i].hist(X[feat].values, bins=50, edgecolor="black", alpha=0.7)
    axes[i].set_title(feat, fontsize=12)
    axes[i].set_ylabel("Count")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
plt.savefig("/tmp/feature_distributions.png", dpi=100, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Correlation Analysis — Phát hiện Collinearity

# COMMAND ----------

# Pearson correlation matrix
corr_matrix = X.corr(method="pearson")

# Heatmap
fig, ax = plt.subplots(figsize=(20, 16))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, cmap="RdBu_r", center=0,
    annot=False, fmt=".2f", linewidths=0.5, ax=ax,
    vmin=-1, vmax=1,
)
ax.set_title("Pearson Correlation Matrix — Candidate Features", fontsize=14)
plt.tight_layout()
plt.savefig("/tmp/correlation_matrix.png", dpi=100, bbox_inches="tight")
plt.show()

# COMMAND ----------

# Phát hiện cặp features có tương quan cao (potential collinearity)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i + 1, len(corr_matrix.columns)):
        corr_val = abs(corr_matrix.iloc[i, j])
        if corr_val >= CORR_THRESHOLD:
            high_corr_pairs.append({
                "feature_1": corr_matrix.columns[i],
                "feature_2": corr_matrix.columns[j],
                "correlation": round(corr_matrix.iloc[i, j], 4),
                "abs_correlation": round(corr_val, 4),
            })

high_corr_df = pd.DataFrame(
    high_corr_pairs,
    columns=["feature_1", "feature_2", "correlation", "abs_correlation"],
).sort_values("abs_correlation", ascending=False)
print(f"\nHighly correlated pairs (|r| >= {CORR_THRESHOLD}): {len(high_corr_df)}")
display(spark.createDataFrame(high_corr_df) if len(high_corr_df) > 0 else print("No highly correlated pairs found."))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Mutual Information — Tương quan phi tuyến

# COMMAND ----------

# MI cho Regression target
mi_reg = mutual_info_regression(X, y_reg, random_state=42, n_neighbors=5)
mi_reg_df = pd.DataFrame({
    "feature": feature_cols,
    "mi_regression": mi_reg,
}).sort_values("mi_regression", ascending=False)

print("=== Mutual Information — Regression Target (target_return_1h) ===")
display(spark.createDataFrame(mi_reg_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Quick LightGBM Feature Importance

# COMMAND ----------

from lightgbm import LGBMRegressor

# --- Regression importance ---
lgbm_reg = LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
lgbm_reg.fit(X, y_reg)
imp_reg = pd.DataFrame({
    "feature": feature_cols,
    "importance_regression": lgbm_reg.feature_importances_,
}).sort_values("importance_regression", ascending=False)

print("=== LightGBM Feature Importance — Regression ===")
display(spark.createDataFrame(imp_reg))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Consolidated Feature Ranking & Selection

# COMMAND ----------

# Gộp tất cả metrics vào một bảng
ranking = pd.DataFrame({"feature": feature_cols})
ranking = ranking.merge(mi_reg_df, on="feature", how="left")
ranking = ranking.merge(imp_reg[["feature", "importance_regression"]], on="feature", how="left")

# Tính rank trung bình (thấp hơn = tốt hơn)
for col in ["mi_regression", "importance_regression"]:
    ranking[f"rank_{col}"] = ranking[col].rank(ascending=False)

rank_cols = [c for c in ranking.columns if c.startswith("rank_")]
ranking["avg_rank"] = ranking[rank_cols].mean(axis=1)
ranking = ranking.sort_values("avg_rank")

print("=== Consolidated Feature Ranking ===")
display(spark.createDataFrame(
    ranking[["feature", "mi_regression", "importance_regression", "avg_rank"]]
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Automatic Feature Selection

# COMMAND ----------

# Bước 1: Loại features có tương quan quá cao (giữ feature có avg_rank thấp hơn)
features_to_drop_corr = set()
for _, row in high_corr_df.iterrows():
    f1, f2 = row["feature_1"], row["feature_2"]
    if f1 in features_to_drop_corr or f2 in features_to_drop_corr:
        continue
    # Giữ lại feature có rank tốt hơn (avg_rank thấp hơn)
    rank_f1 = ranking.loc[ranking["feature"] == f1, "avg_rank"].values
    rank_f2 = ranking.loc[ranking["feature"] == f2, "avg_rank"].values
    if len(rank_f1) > 0 and len(rank_f2) > 0:
        if rank_f1[0] <= rank_f2[0]:
            features_to_drop_corr.add(f2)
        else:
            features_to_drop_corr.add(f1)

print(f"Features dropped due to high correlation: {features_to_drop_corr}")

# Bước 2: Loại features có MI rất thấp với regression target
MI_THRESHOLD = 0.001
low_mi_features = set(
    ranking[ranking["mi_regression"] < MI_THRESHOLD]["feature"].tolist()
)
print(f"Features with very low MI for regression target: {low_mi_features}")

# Bước 3: Tạo danh sách selected features
all_drops = features_to_drop_corr | low_mi_features
selected_features = [f for f in feature_cols if f not in all_drops]

print(f"\n{'='*60}")
print(f"SELECTED FEATURES ({len(selected_features)} / {len(feature_cols)}):")
print(f"{'='*60}")
for f in selected_features:
    print(f"  - {f}")
print(f"\nDropped features ({len(all_drops)}):")
for f in sorted(all_drops):
    print(f"  - {f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Save Selected Features as Config

# COMMAND ----------

created_at = pd.Timestamp.now(tz="UTC")
config_id = int(created_at.timestamp() * 1000)
config_ref = f"{catalog}.{features_schema}.feature_selection_config"

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {config_ref} (
        config_key STRING,
        config_value STRING,
        config_id BIGINT,
        config_version BIGINT,
        created_at STRING,
        created_by STRING,
        is_active BOOLEAN,
        n_features BIGINT,
        method STRING,
        source_table STRING,
        source_table_version BIGINT,
        target_col STRING,
        candidate_features_json STRING,
        dropped_features_json STRING,
        selection_metrics_json STRING,
        corr_threshold DOUBLE,
        mi_threshold DOUBLE
    )
    USING DELTA
""")

for column_spec in [
    "config_id BIGINT",
    "created_by STRING",
    "is_active BOOLEAN",
    "source_table STRING",
    "source_table_version BIGINT",
    "target_col STRING",
    "candidate_features_json STRING",
    "dropped_features_json STRING",
    "selection_metrics_json STRING",
]:
    try:
        spark.sql(f"ALTER TABLE {config_ref} ADD COLUMNS ({column_spec})")
    except Exception as exc:
        print(f"feature config column already exists or cannot be added: {column_spec}; {exc}")

selection_metrics = {
    "top_mi_features": mi_reg_df.head(20).to_dict(orient="records"),
    "top_importance_features": imp_reg.head(20).to_dict(orient="records"),
    "high_corr_pairs": high_corr_df.head(50).to_dict(orient="records") if len(high_corr_df) else [],
}

config_df = spark.createDataFrame([{
    "config_key": "selected_features",
    "config_value": json.dumps(selected_features),
    "config_id": config_id,
    "config_version": config_id,
    "created_at": created_at.isoformat(),
    "created_by": "02b_eda_feature_selection",
    "is_active": True,
    "n_features": len(selected_features),
    "method": "eda_auto_selection",
    "source_table": features_ref,
    "source_table_version": features_table_version,
    "target_col": REGRESSION_TARGET,
    "candidate_features_json": json.dumps(feature_cols),
    "dropped_features_json": json.dumps(sorted(all_drops)),
    "selection_metrics_json": json.dumps(selection_metrics, default=str),
    "corr_threshold": CORR_THRESHOLD,
    "mi_threshold": MI_THRESHOLD,
}])

config_df.write.format("delta").mode("append").option(
    "mergeSchema", "true"
).saveAsTable(config_ref)

spark.sql(f"""
    UPDATE {config_ref}
    SET is_active = false
    WHERE config_key = 'selected_features'
      AND is_active = true
      AND COALESCE(config_id, config_version) != {config_id}
""")

print(f"Selected features config saved to: {config_ref}")
print(f"Feature config id: {config_id}")
print(f"Selected features: {selected_features}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Target Distribution Analysis

# COMMAND ----------

# Phân phối regression target
fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ax.hist(y_reg.values, bins=100, edgecolor="black", alpha=0.7, color="steelblue")
ax.set_title("Distribution: target_return_1h (Regression)", fontsize=12)
ax.set_xlabel("Return")
ax.set_ylabel("Count")
ax.axvline(x=0, color="red", linestyle="--", alpha=0.7)

plt.tight_layout()
plt.savefig("/tmp/target_distributions.png", dpi=100, bbox_inches="tight")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC Notebook này thực hiện:
# MAGIC 1. **Thống kê cơ bản** — kiểm tra null, phân phối
# MAGIC 2. **Correlation analysis** — phát hiện collinearity giữa features
# MAGIC 3. **Mutual Information** — đánh giá tương quan phi tuyến cho regression target
# MAGIC 4. **LightGBM Feature Importance** — quick baseline importance
# MAGIC 5. **Tự động lựa chọn features** — loại features collinear & low-information
# MAGIC 6. **Lưu config** — `selected_features` vào Delta table để `03_optuna_training.py` sử dụng
