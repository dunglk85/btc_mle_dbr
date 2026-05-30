# Kế hoạch triển khai: BTC Databricks MLOps Project

Dự án xây dựng hệ thống MLOps end-to-end trên Databricks cho bài toán dự đoán giá Bitcoin (hourly time series), bao gồm: data pipeline, model training với cơ chế Champion vs Challenger, CI/CD tự động, và monitoring toàn diện.

---

## Tổng quan kiến trúc

```mermaid
graph TB
    subgraph "Data Source"
        A["API/Exchange<br/>(BTC Hourly Data)"]
    end

    subgraph "Databricks Platform (Free Edition)"
        subgraph "Unity Catalog"
            B["Raw Layer<br/>(Delta Table)"]
            C["Features Layer<br/>(Delta Table)"]
        end
        
        subgraph "MLflow"
            D["Experiment Tracking"]
            E["Model Registry"]
            F["Champion Model"]
            G["Challenger Model"]
        end
        
        subgraph "Monitoring"
            H["Data Quality Monitor"]
            I["Model Performance Monitor"]
            J["Job Quality Monitor"]
        end
        
        K["Ingestion Job<br/>(1h)"]
        O["Retrain Job<br/>(3h, config động)"]
        L["Dashboard"]
    end

    subgraph "CI/CD"
        M["GitHub Actions"]
    end

    A -->|"Ingest hourly"| B
    B -->|"Feature Engineering"| C
    C -->|"Train"| D
    D --> G
    G -->|"Compare"| F
    F -->|"Predict next 1h"| N["Prediction Output"]
    M -->|"Deploy & Trigger"| K
    M -->|"Deploy & Trigger"| O
    K --> B
    K --> C
    O --> C
    O --> D
    H --> L
    I --> L
    J --> L
```

---

## Kế hoạch theo tuần

---

### Tuần 1: Setup, Thiết kế & Ingestion

> [!IMPORTANT]
> Tuần này tập trung vào thiết kế kiến trúc, setup môi trường và dựng pipeline nạp dữ liệu thô (Ingestion).

#### Mục tiêu
- Nghiên cứu & quyết định kiến trúc Delta Lake, Unity Catalog, MLflow.
- Setup Git, SP OIDC, và catalog schema (`btc_dev`/`btc_stg`/`btc_prd`).
- Xây dựng Ingestion script tải dữ liệu từ Binance API và lưu Delta Table.

#### Tasks

| # | Task | Chi tiết | Databricks Feature / Tool |
|---|------|----------|-------------------|
| 1.1 | Nghiên cứu Unity Catalog vs DBFS | - Catalog: quản lý metadata tập trung, governance, lineage<br/>- DBFS: file system đơn giản, không có governance<br/>- **Quyết định**: Dùng Unity Catalog cho production | Unity Catalog |
| 1.2 | Nghiên cứu Delta vs Parquet | - Delta: ACID transactions, time travel, schema evolution, merge/upsert<br/>- Parquet: chỉ là format lưu trữ, không có transaction<br/>- **Quyết định**: Dùng Delta cho cả raw và features layer | Delta Lake |
| 1.3 | Thiết kế Catalog schema (multi-env) | - Catalog theo environment: `btc_dev`, `btc_stg`, `btc_prd`<br/>- Schema trong mỗi catalog: `raw`, `features`, `predictions`<br/>- VD: `btc_dev.raw.btc_hourly`<br/>- Tách biệt hoàn toàn giữa các môi trường (cùng 1 workspace, khác catalog) | Unity Catalog |
| 1.4 | Nghiên cứu MLflow trên Databricks | - Experiment tracking, model registry<br/>- Champion/Challenger workflow<br/>- Model versioning & aliases | MLflow |
| 1.5 | Nghiên cứu Databricks CLI | - Setup databricks-cli ở local<br/>- Tạo/quản lý resources bằng terminal<br/>- Databricks Asset Bundles (DABs) | Databricks CLI |
| 1.6 | Setup GitHub repo & branching strategy | - Branch: `main`, `dev`, `feature/*`<br/>- Cấu trúc thư mục project | GitHub |
| 1.7 | Xây dựng Data Ingestion notebook/script | - Kéo dữ liệu BTC hourly từ **Binance API**<br/>- Lưu vào `btc_dev.raw.btc_hourly` (Delta Table)<br/>- Xử lý incremental load (chỉ kéo dữ liệu mới) | Notebooks, Delta Lake |
| 1.8 | Backfill dữ liệu lịch sử | - Kéo dữ liệu từ 2025-01-01 → nay (~12,000+ rows)<br/>- Kiểm tra data quality sau backfill | Delta Lake, SQL |

#### Deliverables
- [x] Tài liệu so sánh Catalog vs DBFS, Delta vs Parquet
- [x] Sơ đồ kiến trúc chi tiết
- [x] Catalog schema design document (multi-env: btc_dev/stg/prd)
- [x] GitHub repo với cấu trúc thư mục chuẩn
- [ ] Ingestion notebook/script hoạt động và backfill thành công

---

### Tuần 2: Feature Engineering & Model Training (Optuna)

#### Mục tiêu
- Xây dựng feature engineering pipeline và đăng ký vào Feature Registry.
- Xây dựng model training pipeline với Optuna hyperparameter tuning (single-node).

#### Tasks

| # | Task | Chi tiết | Databricks Feature |
|---|------|----------|-------------------|
| 2.1 | Feature Engineering pipeline | - Tạo features từ raw data:<br/>  • Moving averages (MA7, MA24, MA168)<br/>  • RSI, MACD, Bollinger Bands<br/>  • Lag features (1h, 2h, 4h, 12h, 24h)<br/>  • Volume features<br/>  • Time-based features (hour, day_of_week)<br/>- Lưu vào `btc_dev.features.btc_features` | Feature Engineering |
| 2.2 | Đăng ký Feature Table | - Register features vào Feature Registry<br/>- Tạo lookup keys và metadata | Feature Store / Feature Registry |
| 2.3 | Data validation | - Kiểm tra null, duplicate, outlier<br/>- Đảm bảo tính liên tục của time series | Data Quality |
| 2.4 | Time Series Data Split | - Temporal split: 80% train, 10% validation, 10% test<br/>- Không random shuffle<br/>- Đảm bảo không data leakage | Python/Spark |
| 2.5 | Training pipeline với Optuna Tuning | - Thuật toán candidates: XGBoost, LightGBM, Random Forest, Linear Regression<br/>- Dùng **Optuna** với TPE Bayesian optimization<br/>- Chạy single-node (phù hợp Free Edition serverless)<br/>- `MedianPruner` early stopping cho trials không triển vọng<br/>- Log tất cả trials vào MLflow (child runs) | MLflow + Optuna |

> [!NOTE]
> **Optuna trên Free Edition**: Free Edition dùng serverless compute, không có PySpark executors thật sự để chạy parallel trials. Dùng `optuna.create_study()` thông thường (single-node) kết hợp MLflow logging thủ công là đủ để demo và hoạt động ổn định.
>
> ```python
> import optuna
> import mlflow
>
> def objective(trial):
>     params = {
>         "n_estimators": trial.suggest_int("n_estimators", 100, 500),
>         "max_depth": trial.suggest_int("max_depth", 3, 10),
>         "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
>     }
>     with mlflow.start_run(nested=True):
>         mlflow.log_params(params)
>         # train & evaluate
>         mlflow.log_metric("rmse", rmse)
>     return rmse
>
> study = optuna.create_study(direction="minimize",
>                             pruner=optuna.pruners.MedianPruner())
> study.optimize(objective, n_trials=50)
> ```

#### Deliverables
- [ ] Feature engineering pipeline & Feature table registered
- [ ] Training pipeline notebook với Optuna tuning hoạt động ổn định

---

### Tuần 3: Model Registry & CI/CD (GitHub Actions + DABs)

> [!IMPORTANT]
> Dữ liệu là time series → **KHÔNG được random split**. Phải dùng temporal split (train trên quá khứ, test trên tương lai).

#### Mục tiêu
- Xây dựng so sánh Champion vs Challenger sử dụng Model Aliases.
- Thiết lập CI/CD pipeline với GitHub Actions và Databricks Asset Bundles (DABs).

#### Tasks

| # | Task | Chi tiết | Databricks Feature / Tool |
|---|------|----------|-------------------|
| 3.1 | Auto-select Best Model (Challenger) | - So sánh metrics (RMSE, MAE, MAPE) trên validation set<br/>- Tự động chọn model tốt nhất → **Challenger**<br/>- Register Challenger vào Model Registry | MLflow Model Registry |
| 3.2 | Champion vs Challenger Comparison | - Load Champion hiện tại từ Model Registry (production alias `@Champion`)<br/>- So sánh Challenger vs Champion trên test set<br/>- Nếu Challenger thắng → promote lên production alias<br/>- Nếu Champion thắng → giữ nguyên | MLflow Model Registry |
| 3.3 | Prediction pipeline | - Model thắng cuộc predict giá BTC cho next 1 hour<br/>- Lưu prediction vào `btc_dev.predictions.btc_predictions` | MLflow, Delta Lake |
| 3.4 | Setup Databricks Asset Bundles (DABs) | - Định nghĩa jobs, notebooks trong `databricks.yml`<br/>- Tách config cho dev/prod bằng DABs `targets` (cùng workspace, khác catalog) | Databricks CLI, DABs |
| 3.5 | GitHub Actions CI/CD Pipeline | - **CI (on PR/Push)**: Lint, unit test, validate bundle<br/>- **CD (on Merge)**: Deploy bundle lên Databricks và cập nhật Jobs | GitHub Actions |
| 3.6 | Environment management | - Secrets: API keys, Databricks tokens<br/>- GitHub Secrets → Databricks Secrets | GitHub Secrets, Databricks Secrets |

> [!NOTE]
> **DABs trên Free Edition**: Free Edition chỉ có 1 workspace. Multi-environment được implement bằng cách dùng DABs `targets` với catalog name khác nhau — không cần nhiều workspace.
>
> ```yaml
> # databricks.yml
> targets:
>   dev:
>     workspace:
>       host: ${DATABRICKS_HOST}
>     variables:
>       catalog: btc_dev
>   prod:
>     workspace:
>       host: ${DATABRICKS_HOST}
>     variables:
>       catalog: btc_prd
> ```
> Deploy dev: `databricks bundle deploy --target dev`
> Deploy prod: `databricks bundle deploy --target prod`

#### Workflow Champion vs Challenger

```mermaid
flowchart TD
    A["Feature Data Ready"] --> B["Optuna Tuning<br/>(50 trials, TPE Sampler)"]
    B --> C["Select Best → Challenger"]
    D{"Champion exists?"}
    C --> D
    D -->|No| E["Promote Challenger<br/>→ Champion"]
    D -->|Yes| F["Compare on Test Set"]
    F --> G{"Challenger better?"}
    G -->|Yes| H["Promote Challenger<br/>→ New Champion"]
    G -->|No| I["Keep Current Champion"]
    H --> J["Predict Next 1h"]
    I --> J
    E --> J
    J --> K["Save Prediction<br/>to Delta Table"]
```

#### Metrics đánh giá

| Metric | Mô tả | Mục đích |
|--------|--------|----------|
| RMSE | Root Mean Squared Error | Đánh giá tổng thể |
| MAE | Mean Absolute Error | Dễ diễn giải |
| MAPE | Mean Absolute Percentage Error | So sánh tương đối |
| R² | Coefficient of Determination | Giải thích variance |
| Time to tune | Training time for all trials | Đánh giá hiệu quả tuning |
| n_trials / early_stopped | Số trials chạy / bị dừng sớm | Đo mức độ hội tụ của Optuna |

#### Deliverables
- [ ] Champion vs Challenger comparison logic & prediction pipeline
- [ ] Databricks Asset Bundle config & GitHub Actions CI/CD workflows hoạt động

---

### Tuần 4: Deploy & Monitoring

#### Mục tiêu
- Thiết lập Databricks Workflows chạy định kỳ.
- Xây dựng AI/BI Dashboard và hệ thống cảnh báo tự động.

#### Tasks

| # | Task | Chi tiết | Databricks Feature |
|---|------|----------|-------------------|
| 4.1 | Databricks Jobs (Schedule) | - **Ingestion Job** (chạy 1h/lần): Ingest + Feature Engineering + cập nhật `actual_close` cho predictions cũ<br/>- **Retrain Job** (chạy 3h/lần, config động): Model Training + Champion vs Challenger + Prediction<br/>- Cấu hình retry, timeout, alerts | Databricks Jobs/Workflows |
| 4.2 | Data Quality Monitoring | - Theo dõi: missing values, schema drift, data freshness<br/>- Kiểm tra tính liên tục của time series<br/>- Alert khi data bất thường | Databricks Data Quality / Lakehouse Monitoring |
| 4.3 | Model Performance Monitoring | - Theo dõi: prediction accuracy theo thời gian, model drift/data drift<br/>- So sánh actual vs predicted (dùng `actual_close` đã được cập nhật từ Ingestion Job)<br/>- Theo dõi Optuna trial history (convergence plot, parameter importance)<br/>- Alert khi performance giảm | MLflow, Lakehouse Monitoring |
| 4.4 | Job Quality Monitoring | - Theo dõi: job success/failure rate, duration<br/>- Alert khi job fail hoặc chạy quá lâu | Databricks Jobs, Alerts |
| 4.5 | Tạo Dashboard | - Tổng hợp tất cả metrics monitoring<br/>- Hiển thị: data freshness, model accuracy trend, job status, biểu đồ actual vs predicted price | Databricks Dashboard (Lakeview) |
| 4.6 | Thiết lập Alerts | - Email/Slack notification khi job fail, data quality issue hoặc model performance drop | Databricks Alerts |

#### Dashboard mockup

```
┌─────────────────────────────────────────────────────────┐
│                 BTC MLOps Dashboard                     │
├───────────────┬───────────────┬─────────────────────────┤
│ Data Quality  │ Model Perf    │ Job Status              │
│ ✅ Fresh      │ RMSE: 125.3   │ ✅ Last run: 14:00 OK   │
│ ✅ Complete   │ MAE: 89.2     │ Success rate: 99.2%     │
│ ✅ No drift   │ MAPE: 0.12%   │ Avg duration: 3m 42s    │
├───────────────┴───────────────┴─────────────────────────┤
│          Actual vs Predicted Price (Last 7 days)        │
│  📈 [Chart]                                             │
├─────────────────────────────────────────────────────────┤
│          Model Performance Trend (Last 30 days)         │
│  📊 [Chart]                                             │
└─────────────────────────────────────────────────────────┘
```

#### Deliverables
- [ ] Ingestion Job & Retrain Job hoạt động ổn định trên Databricks
- [ ] Data, Model & Job Quality monitoring setup
- [ ] Dashboard hoàn chỉnh & Alert configuration

---

## Cấu trúc thư mục dự án

```
BTC/
├── .github/
│   └── workflows/
│       ├── ci.yml
│       ├── cd.yml
│       └── hourly-trigger.yml
├── databricks/
│   ├── databricks.yml          # DABs config với targets (dev/prod)
│   └── resources/
│       └── jobs.yml            # Job definitions
├── notebooks/
│   ├── 01_data_ingestion.py        # Kéo dữ liệu BTC + cập nhật actual_close
│   ├── 02_feature_engineering.py   # Tạo features
│   ├── 03_model_training.py        # Train + Optuna Tuning
│   ├── 04_champion_challenger.py   # So sánh models
│   └── 05_prediction.py            # Predict next 1h
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   └── features.py
│   ├── models/
│   │   ├── training.py
│   │   └── evaluation.py
│   ├── monitoring/
│   │   ├── data_quality.py
│   │   └── model_performance.py
│   └── utils/
│       ├── config.py
│       └── logger.py
├── tests/
│   ├── test_ingestion.py
│   ├── test_features.py
│   └── test_training.py
├── configs/
│   └── optuna_params.py       # Search space config cho từng model
├── docs/
│   ├── btc-databricks-mlops-project.md
│   ├── architecture.md
│   └── catalog-schema.md
└── requirements.txt
```

> [!NOTE]
> `configs/dev.yml`, `staging.yml`, `prod.yml` đã được gộp vào `databricks/databricks.yml` theo DABs `targets` pattern. Không cần file config env riêng lẻ.

---

## Databricks Features Mapping

| Nhiệm vụ | Databricks Feature | Ghi chú |
|-----------|-------------------|---------|
| Lưu trữ dữ liệu | **Unity Catalog** + **Delta Lake** | Vẫn dùng được trên Free edition |
| Feature Engineering | **Feature Store / Feature Registry** | Quản lý features tập trung |
| Train model | **MLflow Experiments** | Auto-log params, metrics, artifacts |
| Model Registry | **MLflow Model Registry** | Champion/Challenger với aliases |
| Hyperparameter Tuning | **Optuna** + **MLflow** | TPE Bayesian optimization, single-node, early stopping |
| Chạy job | **Databricks Jobs/Workflows** | 2 jobs: Ingestion (1h) + Retrain (3h, config động) |
| CI/CD | **Databricks Asset Bundles (DABs)** | IaC cho Databricks resources, multi-env qua targets |
| Data Quality | **Lakehouse Monitoring** | Tự động detect anomalies |
| Model Monitoring | **Lakehouse Monitoring** + **MLflow** | Drift detection, actual vs predicted |
| Dashboard | **Lakeview Dashboard** | SQL-based dashboards |
| Alerts | **Databricks Alerts** | Email/Slack notifications |
| CLI management | **Databricks CLI** | Tạo/quản lý resources từ local |

---

## Tất cả quyết định đã được thống nhất ✅

> [!NOTE]
> ✅ **Data Source API**: Binance API
> ✅ **Databricks Workspace**: Free edition
> ✅ **Databricks tier**: Free edition (vẫn dùng Unity Catalog)
> ✅ **DBR version**: 17.3 LTS ML (LTS mới nhất, supported đến 10/2028)
> ✅ **Target variable**: close price (regression)
> ✅ **Retraining**: default 3h, config động
> ✅ **Alert channels**: Email + Slack
> ✅ **Optuna**: single-node (phù hợp Free Edition serverless, không dùng MlflowSparkStudy)
> ✅ **Multi-env**: DABs targets với catalog name khác nhau (btc_dev / btc_prd) trong cùng 1 workspace

> [!NOTE]
> ℹ️ **Compute**: Free Edition dùng serverless compute, không có cluster cost. Databricks tự quản lý, chỉ cần theo dõi daily quota.

---

## Verification Plan

### Automated Tests
- Unit tests cho data ingestion, feature engineering, model training
- Integration test: chạy full pipeline trên sample data
- Validate Delta Table schema sau mỗi bước

### Manual Verification
- Kiểm tra dữ liệu trên Databricks UI (Catalog Explorer)
- Review MLflow experiments & model registry
- Kiểm tra Dashboard hiển thị đúng metrics
- Chạy thử Ingestion job (1h) và Retrain job (3h), confirm predictions
- Kiểm tra `actual_close` được cập nhật đúng sau mỗi giờ
