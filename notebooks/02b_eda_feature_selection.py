# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02b - EDA & Feature Selection
# MAGIC
# MAGIC Notebook phân tích khám phá dữ liệu (EDA) và lựa chọn features tối ưu
# MAGIC cho cả **Regression** (`target_return_1h`) và **Classification** (`target_direction_1h`).
# MAGIC
# MAGIC **Phương pháp:**
# MAGIC 1. Kiểm tra phân phối & thống kê cơ bản các features
# MAGIC 2. Correlation Matrix (Pearson & Spearman) — phát hiện collinearity
# MAGIC 3. Mutual Information — đánh giá tương quan phi tuyến với cả hai targets
# MAGIC 4. ANOVA F-value — cho Classification target
# MAGIC 5. Feature Importance sơ bộ bằng LightGBM
# MAGIC 6. Xuất danh sách `selected_features` cuối cùng

# COMMAND ----------

# MAGIC %pip install lightgbm scikit-learn matplotlib seaborn

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.feature_selection import f_classif
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
CLASSIFICATION_TARGET = "target_direction_1h"

# Load data
source = spark.table(features_ref).orderBy("open_time")
all_cols = ["open_time", REGRESSION_TARGET, CLASSIFICATION_TARGET] + CANDIDATE_FEATURES
# Chỉ lấy các cột thực sự có trong bảng
existing_cols = [c for c in all_cols if c in source.columns]
missing_cols = [c for c in all_cols if c not in source.columns]
if missing_cols:
    print(f"WARNING: Missing columns (skipped): {missing_cols}")

pdf = source.select(existing_cols).toPandas().sort_values("open_time").reset_index(drop=True)

# Loại bỏ rows có target null (dòng cuối cùng do shift)
pdf = pdf.dropna(subset=[REGRESSION_TARGET, CLASSIFICATION_TARGET])
print(f"Total rows for EDA: {len(pdf)}")

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
pdf_clean = pdf[feature_cols + [REGRESSION_TARGET, CLASSIFICATION_TARGET]].dropna()
print(f"Rows after dropping NaN: {len(pdf_clean)} (dropped {len(pdf) - len(pdf_clean)} rows)")

X = pdf_clean[feature_cols]
y_reg = pdf_clean[REGRESSION_TARGET]
y_cls = pdf_clean[CLASSIFICATION_TARGET].astype(int)

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

high_corr_df = pd.DataFrame(high_corr_pairs).sort_values("abs_correlation", ascending=False)
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

# MI cho Classification target
mi_cls = mutual_info_classif(X, y_cls, random_state=42, n_neighbors=5)
mi_cls_df = pd.DataFrame({
    "feature": feature_cols,
    "mi_classification": mi_cls,
}).sort_values("mi_classification", ascending=False)

print("=== Mutual Information — Classification Target (target_direction_1h) ===")
display(spark.createDataFrame(mi_cls_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. ANOVA F-value — cho Classification

# COMMAND ----------

f_scores, p_values = f_classif(X, y_cls)
anova_df = pd.DataFrame({
    "feature": feature_cols,
    "f_score": f_scores,
    "p_value": p_values,
}).sort_values("f_score", ascending=False)

print("=== ANOVA F-value — Classification Target ===")
display(spark.createDataFrame(anova_df))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Quick LightGBM Feature Importance

# COMMAND ----------

from lightgbm import LGBMRegressor, LGBMClassifier

# --- Regression importance ---
lgbm_reg = LGBMRegressor(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
lgbm_reg.fit(X, y_reg)
imp_reg = pd.DataFrame({
    "feature": feature_cols,
    "importance_regression": lgbm_reg.feature_importances_,
}).sort_values("importance_regression", ascending=False)

# --- Classification importance ---
lgbm_cls = LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)
lgbm_cls.fit(X, y_cls)
imp_cls = pd.DataFrame({
    "feature": feature_cols,
    "importance_classification": lgbm_cls.feature_importances_,
}).sort_values("importance_classification", ascending=False)

print("=== LightGBM Feature Importance — Regression ===")
display(spark.createDataFrame(imp_reg))

print("=== LightGBM Feature Importance — Classification ===")
display(spark.createDataFrame(imp_cls))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Consolidated Feature Ranking & Selection

# COMMAND ----------

# Gộp tất cả metrics vào một bảng
ranking = pd.DataFrame({"feature": feature_cols})
ranking = ranking.merge(mi_reg_df, on="feature", how="left")
ranking = ranking.merge(mi_cls_df, on="feature", how="left")
ranking = ranking.merge(anova_df[["feature", "f_score", "p_value"]], on="feature", how="left")
ranking = ranking.merge(imp_reg[["feature", "importance_regression"]], on="feature", how="left")
ranking = ranking.merge(imp_cls[["feature", "importance_classification"]], on="feature", how="left")

# Tính rank trung bình (thấp hơn = tốt hơn)
for col in ["mi_regression", "mi_classification", "f_score", "importance_regression", "importance_classification"]:
    ranking[f"rank_{col}"] = ranking[col].rank(ascending=False)

rank_cols = [c for c in ranking.columns if c.startswith("rank_")]
ranking["avg_rank"] = ranking[rank_cols].mean(axis=1)
ranking = ranking.sort_values("avg_rank")

print("=== Consolidated Feature Ranking ===")
display(spark.createDataFrame(
    ranking[["feature", "mi_regression", "mi_classification", "f_score",
             "importance_regression", "importance_classification", "avg_rank"]]
))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Automatic Feature Selection

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

# Bước 2: Loại features có MI rất thấp cho CẢ HAI targets
MI_THRESHOLD = 0.001
low_mi_features = set(
    ranking[
        (ranking["mi_regression"] < MI_THRESHOLD) &
        (ranking["mi_classification"] < MI_THRESHOLD)
    ]["feature"].tolist()
)
print(f"Features with very low MI for both targets: {low_mi_features}")

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
# MAGIC ## 10. Save Selected Features as Config

# COMMAND ----------

# Lưu selected_features vào Delta table để notebook training có thể đọc
import json

config_df = spark.createDataFrame([{
    "config_key": "selected_features",
    "config_value": json.dumps(selected_features),
    "created_at": pd.Timestamp.now().isoformat(),
    "n_features": len(selected_features),
    "method": "eda_auto_selection",
    "corr_threshold": CORR_THRESHOLD,
    "mi_threshold": MI_THRESHOLD,
}])

config_ref = f"{catalog}.{features_schema}.feature_selection_config"
config_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(config_ref)

print(f"Selected features config saved to: {config_ref}")
print(f"Selected features: {selected_features}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Target Distribution Analysis

# COMMAND ----------

# Phân phối classification target — kiểm tra class imbalance
class_dist = y_cls.value_counts()
print(f"Classification target distribution:")
print(f"  Class 0 (down/flat): {class_dist.get(0, 0)} ({class_dist.get(0, 0)/len(y_cls)*100:.1f}%)")
print(f"  Class 1 (up):        {class_dist.get(1, 0)} ({class_dist.get(1, 0)/len(y_cls)*100:.1f}%)")
print(f"  Imbalance ratio:     {max(class_dist)/min(class_dist):.2f}")

# Phân phối regression target
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(y_reg.values, bins=100, edgecolor="black", alpha=0.7, color="steelblue")
axes[0].set_title("Distribution: target_return_1h (Regression)", fontsize=12)
axes[0].set_xlabel("Return")
axes[0].set_ylabel("Count")
axes[0].axvline(x=0, color="red", linestyle="--", alpha=0.7)

class_dist.plot.bar(ax=axes[1], color=["salmon", "steelblue"], edgecolor="black")
axes[1].set_title("Distribution: target_direction_1h (Classification)", fontsize=12)
axes[1].set_xlabel("Class")
axes[1].set_ylabel("Count")
axes[1].set_xticklabels(["0 (Down/Flat)", "1 (Up)"], rotation=0)

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
# MAGIC 3. **Mutual Information** — đánh giá tương quan phi tuyến cho cả Regression & Classification
# MAGIC 4. **ANOVA F-test** — cho Classification
# MAGIC 5. **LightGBM Feature Importance** — quick baseline importance
# MAGIC 6. **Tự động lựa chọn features** — loại features collinear & low-information
# MAGIC 7. **Lưu config** — `selected_features` vào Delta table để `03_optuna_training.py` sử dụng
