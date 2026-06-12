# Architecture

## Tổng Quan Kiến Trúc

BTC Databricks MLOps là pipeline dự đoán giá Bitcoin theo chu kỳ hourly, chạy trên Databricks Asset Bundles và Unity Catalog. Kiến trúc hiện tại gộp inference và training vào một job duy nhất: `btc_inference_job` chạy theo giờ, và training chỉ thực thi khi drift vượt ngưỡng (conditional gate).

```mermaid
graph TB
    subgraph "Data Source"
        A["Binance Vision API<br/>(BTCUSDT 1h closed klines)"]
    end

    subgraph "Databricks Platform"
        subgraph "Unity Catalog"
            B["raw.btc_hourly<br/>(Delta Table)"]
            C["features.btc_features<br/>(Delta Table)"]
            U["features.feature_selection_config<br/>(Delta Table)"]
            P["predictions.btc_predictions<br/>(Delta Table)"]
            Q["monitoring.pipeline_metrics<br/>(Delta Table)"]
            T["monitoring.training_dataset_manifests<br/>(Delta Table)"]
            X["monitoring.model_explanations<br/>(Delta Table)"]
        end

        subgraph "MLflow"
            D["Experiment Tracking<br/>(LightGBM + XGBoost + Random Forest runs)"]
            E["UC Model Registry<br/>models.btc_price_model"]
            F["@Champion"]
            G["@Challenger"]
        end

        subgraph "Inference Job (hourly)"
            K["btc_inference_job<br/>(hourly)"]
            K1["01_data_ingestion"]
            K2["02_feature_engineering<br/>(features + selected_features config)"]
            K8["05_prediction"]
            K9["06_monitoring<br/>(drift metrics + task value)"]
            K11["check_drift_threshold<br/>(condition_task)"]
            K4["03_optuna_training<br/>(conditional: LGBM/XGB/RF)"]
            K7["04_champion_challenger<br/>(conditional: select best + promote)"]
        end
    end

    subgraph "CI/CD"
        M["GitHub Actions<br/>(CI/CD)"]
        N["Databricks Asset Bundles<br/>(dev/prod catalog variable)"]
    end

    A -->|"Fetch closed candles"| K1
    K1 -->|"Validate + dedupe + MERGE by open_time"| B
    B -->|"Feature engineering + exact target_close_1h"| K2
    K2 --> C
    K2 -->|"Auto selected_features config"| U
    C -->|"Train on latest features"| K4
    U -->|"Active feature config"| K4
    K4 --> D
    K4 -->|"Write dataset manifest"| T
    K4 -->|"SHAP / explanations"| X
    K4 -->|"Candidate metrics + task values"| K7
    K7 --> G
    G -->|"Bounded fair RMSE/MAE comparison"| F
    F -->|"Promote / retain"| E
    E -->|"Champion model"| K8
    C -->|"Latest feature row"| K8
    K8 --> P
    B --> K9
    C --> K9
    P --> K9
    K9 --> Q
    K9 -->|"Optional drift trigger"| K10
    M --> N
    N --> K
    K --> K1
    K1 --> K2
    K2 --> K8
    K8 --> K9
    K9 --> K11
    K11 -->|"drift ≥ threshold"| K4
    K4 --> D
    K4 -->|"Write dataset manifest"| T
    K4 -->|"SHAP / explanations"| X
    K4 -->|"Candidate metrics + task values"| K7
    K7 --> G
    G -->|"Bounded fair RMSE/MAE comparison"| F
    F -->|"Promote / retain"| E
    E -->|"Champion model"| K8
```

Mỗi lần chạy inference job sẽ lấy nến BTC hourly đã đóng từ Binance Vision API, ghi trực tiếp vào raw Delta table, rebuild feature table, tạo prediction bằng Champion hiện tại, sau đó ghi monitoring và drift metrics. Nếu drift alert count đạt ngưỡng, job sẽ tự động chạy training (LightGBM, XGBoost, Random Forest), chọn challenger tốt nhất, rồi promotion Champion/Challenger nếu đạt điều kiện — tất cả trong cùng một job run.

Các lớp dữ liệu chính:
- `raw.btc_hourly`: dữ liệu OHLCV hourly từ Binance.
- `features.btc_features`: feature table và target next-hour.
- `features.feature_selection_config`: cấu hình feature active dùng cho training.
- `predictions.btc_predictions`: prediction output kèm lineage.
- `monitoring.*`: metrics, manifests, explanations và audit tables.
- `models.btc_price_model`: UC registered model với alias `@Champion` và `@Challenger`.

Thiết kế ưu tiên serving path ngắn và chi phí thấp: không còn UC Volume landing hay Auto Loader staging. Training được gộp vào inference job nhưng chỉ thực thi khi drift vượt ngưỡng qua `condition_task`, tránh lãng phí compute khi model vẫn ổn định.

## Data Flow

1. **Direct Binance ingestion** -> `01_data_ingestion` fetches closed BTC hourly candles from Binance Vision API and MERGEs them into `<catalog>.raw.btc_hourly`.
2. **Feature Engineering + Selection** -> `02_feature_engineering` writes `<catalog>.features.btc_features` with exact next-hour target `target_close_1h` and updates active selected-feature metadata in `<catalog>.features.feature_selection_config`.
3. **Prediction** -> `<catalog>.predictions.btc_predictions` using `@Champion`; return forecasts are converted to `predicted_close` for monitoring.
4. **Monitoring** -> `<catalog>.monitoring.pipeline_metrics`, pipeline metrics, drift metrics, and drift alert count task value for conditional training gate.
5. **Model Training (conditional)** -> On-demand regression-only Optuna LightGBM/XGBoost/Random Forest training + MLflow tracking, triggered only when drift alert count exceeds threshold.
6. **Champion vs Challenger** -> Select the best candidate run, register it as Challenger, evaluate Challenger and current Champion on the same bounded holdout rows, then promote only if RMSE and MAE improve and directional accuracy does not regress.

## Multi-Environment

| Environment | Unity Catalog | DABs Target |
|-------------|---------------|-------------|
| Simplying   | `btc_simply`     | `simplying` |
| Production  | `btc_prod`    | `prod`      |

## Schedules

- **Inference job**: every hour; runs ingestion, feature engineering, Champion prediction, and monitoring.
- **Training tasks**: conditional within inference job; only execute when drift alert count ≥ threshold via `check_drift_threshold` condition task.

## Environment Parameterization

Databricks notebooks read the `catalog` widget passed by Databricks Asset Bundles. The `simplying` target passes `btc_simply`; the `prod` target passes `btc_prod`.

## Data Correctness Rules

- Fetch excludes currently open candles by requiring Binance `close_time` to be before current UTC time.
- Feature target is an exact one-hour lookup, not just the next available row.
- Feature selection config is append-only with one active config; each config records source feature table version, target column, candidates, dropped features, and selection metrics.
- Ingestion reads from the latest raw `open_time` by default and can backfill from a `start_date` widget.
- Ingestion deduplicates overlapping Binance candles by `open_time` before MERGE.
- Training logs Delta versions for raw/features/config tables into MLflow and `monitoring.training_dataset_manifests`.
- Champion/Challenger evaluation uses a bounded latest common holdout window so both models are compared on identical rows without loading the full feature table.
- Predictions store model version/run ID, prediction-input raw/features Delta versions, and Champion training data/config versions for traceability.

## Drift Monitoring Status

Current monitoring is operational fallback monitoring, not full statistical drift detection.

Implemented now:
- Raw freshness.
- Feature row count and target null checks.
- Prediction availability and age.
- Actual-vs-predicted SQL queries for dashboard/alerts.
- Data drift: PSI and approximate KS for `return_1h`.
- Label drift: PSI/KS for `target_close_1h`.
- Prediction drift: PSI/KS for `predicted_close`.
- Model/performance drift: rolling RMSE, MAE, MAPE, R², p95 absolute error, direction accuracy.
- Concept drift proxy: rolling signed error bias.

Conditional training gate:
- Monitoring sets `drift_alert_count` task value.
- `check_drift_threshold` condition task evaluates `drift_alert_count >= 2`.
- If true: training tasks (LGBM, XGB, RF) run → champion_challenger runs.
- If false: training tasks are skipped, job completes successfully.

Job structure:
- `btc_inference_job` runs hourly: ingestion, feature engineering, Champion prediction, monitoring, and conditional training + champion/challenger promotion.
- Training is decoupled from hourly inference via `condition_task` to avoid retraining on the full feature table every hour.
