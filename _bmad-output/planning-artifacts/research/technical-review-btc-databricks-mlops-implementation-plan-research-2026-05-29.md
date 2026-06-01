---
stepsCompleted: [1, 2, 3, 4, 5, 6]
inputDocuments:
  - _bmad-output/planning-artifacts/implementation_plan.md
workflowType: 'research'
lastStep: 6
research_type: 'technical'
research_topic: 'Review kế hoạch triển khai BTC Databricks MLOps Project'
user_name: 'Admin'
date: '2026-05-29'
---

# Báo cáo Nghiên cứu Kỹ thuật: Đánh giá Kế hoạch Triển khai BTC Databricks MLOps

Báo cáo này đánh giá tính khả thi của kế hoạch triển khai hệ thống MLOps dự báo giá BTC theo giờ trên **Databricks Free Edition**, tập trung vào các rủi ro kỹ thuật có thể làm chặn triển khai: network egress, quota compute, Model Registry trên Unity Catalog, CI/CD authentication, scheduling và monitoring.

> [!IMPORTANT]
> Một số kết luận trong bản kế hoạch cần được xác minh trực tiếp trên workspace Free Edition đang dùng. Network spike A1 đã được xác nhận **không chạy được** khi gọi Binance qua `python-binance` từ Databricks, nên ingestion trực tiếp từ Databricks không còn là phương án chính. Spike A2 đã **passed**, xác nhận Unity Catalog Volume/Table dùng được trong workspace hiện tại. Spike A3 đã **passed**, xác nhận MLflow Model Aliases hoạt động với Unity Catalog model. Với A4, dự án quyết định dùng Databricks token/PAT cho CI/CD thay vì OIDC.

---

## 1. Phát hiện Kỹ thuật Cốt lõi & Khuyến nghị

### 1.1. Giới hạn mạng (Outbound Block)
* **Kết quả spike:** **Failed** - Databricks không gọi được Binance qua `python-binance` trong môi trường hiện tại.
* **Quyết định:** Không triển khai ingestion trực tiếp từ Databricks notebook/job.
* **Phương án chính:** Dùng runner trung gian độc lập như GitHub Actions Runner hoặc AWS Lambda free tier để tải dữ liệu Binance bằng `python-binance`, đóng gói CSV/Parquet, rồi upload vào Unity Catalog Volume bằng Databricks CLI hoặc REST API.
* **Tác động kiến trúc:** Databricks nhận dữ liệu đã được staging vào UC Volume/Delta table; job trong Databricks chỉ làm load/merge, feature engineering, training, prediction và monitoring.

### 1.2. Giới hạn tài nguyên (Compute Quotas)
* **Rủi ro:** Optuna 50 trials với nhiều thuật toán candidate có thể vượt quota hoặc làm job retrain không ổn định trên Free Edition.
* **Khuyến nghị:** Đặt default `n_trials=15`, `max_trials=20`, dùng `MedianPruner`, cấu hình timeout theo job và log số trial bị prune vào MLflow.
* **Điều chỉnh lịch retrain:** Bắt đầu với retrain **12h/lần** trong giai đoạn đầu. Chỉ giảm xuống 6h hoặc 3h sau khi đo được duration, quota consumption và độ ổn định job.
* **Fallback:** Nếu quota vẫn thiếu, tách training thủ công/on-demand khỏi prediction hourly; ingestion và prediction vẫn chạy định kỳ, retrain chạy ít hơn.

### 1.3. Quản lý mô hình (Model Registry)
* **Rủi ro:** Stage-based Model Registry (`Staging`/`Production`) không còn là hướng triển khai phù hợp cho Unity Catalog.
* **Khuyến nghị:** Dùng **Model Aliases** như `@Champion` và `@Challenger` với registered model dạng `catalog.schema.model_name`.
* **Kết quả spike:** **Passed** - register model vào Unity Catalog, set alias `Champion` và load lại model bằng alias thành công.
* **Quyết định:** Khóa thiết kế Champion/Challenger bằng MLflow Model Aliases.
* **Cách thao tác:** Promotion/demotion nên thực hiện bằng MLflow Python API hoặc REST API. SQL chủ yếu dùng cho quản trị Unity Catalog như `GRANT`, ownership và schema/table governance.
* **Đường dẫn load model:** `models:/<catalog>.<schema>.<model_name>@Champion`.

### 1.4. Xác thực CI/CD bảo mật
* **Quyết định:** Dùng Databricks token/PAT cho GitHub Actions CI/CD thay vì OIDC.
* **Rủi ro:** PAT dài hạn trong GitHub Secrets dễ tạo rủi ro lộ lọt và khó kiểm soát vòng đời.
* **Khuyến nghị:** Dùng token có thời hạn ngắn nhất có thể, cấp quyền tối thiểu, lưu trong GitHub Environments/Secrets và bật protection cho môi trường deploy.
* **Biện pháp giảm rủi ro:** Tách token dev/prod nếu có thể, đặt lịch rotation, không log token trong workflow, chỉ dùng token cho `databricks bundle validate/deploy` và upload file cần thiết.

### 1.5. Monitoring & Alerting
* **Rủi ro:** Lakehouse Monitoring, Databricks SQL Alerts, Slack/email notifications hoặc AI/BI Dashboard có thể có giới hạn theo edition/workspace.
* **Kết quả hiện tại:** **Deferred** - chưa xử lý A6 trong giai đoạn đầu.
* **Khuyến nghị:** Thiết kế monitoring theo 2 tầng: tầng managed feature nếu khả dụng, và tầng fallback bằng Delta monitoring tables + SQL dashboard thủ công.
* **Fallback tối thiểu:** Ghi `job_run_metrics`, `data_quality_metrics`, `prediction_metrics` vào Delta tables; dashboard đọc trực tiếp các bảng này; alert ban đầu dựa vào Databricks Job notifications hoặc GitHub Actions failure notification.

### 1.6. Unity Catalog Storage
* **Kết quả spike:** **Passed** - Unity Catalog Volume/Table hoạt động trong workspace Free Edition hiện tại.
* **Quyết định:** Dùng UC Volume làm landing/staging area cho file được upload từ runner trung gian, sau đó dùng Delta Table làm raw/features/predictions layer.
* **Tác động kiến trúc:** GitHub Actions/Lambda chỉ chịu trách nhiệm lấy dữ liệu và upload file; Databricks chịu trách nhiệm validate, deduplicate, merge và quản trị dữ liệu bằng Unity Catalog.

### 1.7. Databricks Asset Bundles Multi-target
* **Kết quả hiện tại:** **Deferred / tạm fail giai đoạn đầu** - chưa xử lý DAB targets `dev`/`prod` ở Tuần 1-2.
* **Quyết định:** Không để A5 chặn ingestion, feature engineering và training ban đầu.
* **Phương án giai đoạn đầu:** Dùng một target `dev` hoặc deploy thủ công tối thiểu trước để hoàn thiện pipeline end-to-end.
* **Thời điểm xử lý:** Đưa DAB multi-target (`dev`/`prod`) sang Tuần 3 khi bắt đầu phần Model Registry & CI/CD.

### 1.8. Optuna Quota Validation
* **Kết quả hiện tại:** **Deferred** - chưa xử lý A7 trong giai đoạn đầu.
* **Quyết định:** Không để Optuna quota spike chặn việc xây dựng pipeline training cơ bản.
* **Phương án giai đoạn đầu:** Train baseline model trước, sau đó mới bật Optuna với số trials thấp và timeout rõ ràng.
* **Thời điểm xử lý:** Đưa A7 sang sau khi có feature table và training pipeline tối thiểu.

---

## 2. Bảng so sánh Registry: Stage-based vs Alias-based

| Tiêu chí | Stage-based (Cũ) | Alias-based (Hiện tại - Unity Catalog) |
|---|---|---|
| **Cơ chế** | Chuyển Stage cứng (`Staging`, `Production`) | Sử dụng các thẻ tên động (Aliases như `@Champion`, `@Challenger`) |
| **Bảo mật** | Phân quyền mức Workspace rộng | Phân quyền 3 cấp bằng SQL GRANT trong Unity Catalog |
| **Đường dẫn** | `models:/<model_name>/production` | `models:/<catalog>.<schema>.<model_name>@Champion` |
| **Khuyến nghị** | Đã bị loại bỏ (Deprecated) | Khuyến nghị sử dụng chính thức |

---

## 3. Assumption Validation Checklist

Các giả định sau cần được xác minh trong 1-2 ngày đầu trước khi triển khai sâu. Nếu một giả định fail, áp dụng fallback tương ứng thay vì tiếp tục theo thiết kế ban đầu.

| # | Giả định cần xác minh | Cách kiểm tra | Kết quả mong muốn | Fallback nếu fail |
|---|---|---|---|---|
| A1 | Databricks notebook/job gọi được Binance API qua `python-binance` | Notebook cài/import `python-binance`, gọi `Client.get_historical_klines()` hoặc `Client.get_klines()` và ghi response | **Failed trong workspace hiện tại** | GitHub Actions/Lambda ingest bằng `python-binance` rồi upload UC Volume |
| A2 | Unity Catalog Volume/Table hoạt động trong Free Edition | Tạo catalog/schema/volume/table test | **Passed** - create/read/write thành công | Không cần fallback hiện tại |
| A3 | MLflow Model Alias hoạt động với UC model | Register model, set alias `Champion`, load bằng alias | **Passed** - load `models:/catalog.schema.model@Champion` thành công | Không cần fallback hiện tại |
| A4 | GitHub Actions deploy Databricks bằng token/PAT | GitHub Actions dùng `DATABRICKS_HOST` và `DATABRICKS_TOKEN` để chạy `databricks bundle validate/deploy` | **Decision: dùng token/PAT** | Giảm rủi ro bằng token ngắn hạn, rotation và protected environment |
| A5 | DAB targets dev/prod dùng được trong cùng workspace | Deploy `dev` và `prod` vào catalog khác nhau | **Deferred / tạm fail giai đoạn đầu** - xử lý ở Tuần 3 | Một target `dev` trước, prod deploy thủ công sau |
| A6 | Monitoring/alerts đủ tính năng | Tạo dashboard, job notification, SQL alert thử | **Deferred** - xử lý sau | Delta metrics tables + job/GitHub notifications |
| A7 | Optuna không vượt quota | Chạy 5, 10, 15 trials và đo duration/quota | **Deferred** - xử lý sau khi có training pipeline cơ bản | Giảm trials, retrain thưa hơn, training on-demand |

---

## 4. Lộ trình Triển khai & Quản lý Rủi ro

### Risk Gate Trước Tuần 1

Trước khi bắt đầu build đầy đủ, cần hoàn tất các spike sau:

1. **Network spike:** Đã xác minh Databricks không gọi được Binance API qua `python-binance`; chọn ingestion fallback qua runner trung gian.
2. **Registry spike:** Đã passed; dùng MLflow Model Aliases cho Champion/Challenger.
3. **CI/CD auth:** Dùng Databricks token/PAT trong GitHub Actions, ưu tiên token ngắn hạn và protected environment.
4. **DAB multi-target:** Deferred sang Tuần 3; không chặn Tuần 1-2.
5. **Monitoring spike:** Deferred; không chặn giai đoạn ingestion/training cơ bản.
6. **Optuna quota spike:** Deferred; chạy sau khi có feature table và baseline training.

Chỉ khi các spike này có kết quả rõ ràng mới khóa kiến trúc cuối cùng cho ingestion, deployment và monitoring.

```mermaid
gantt
    title Lộ trình triển khai BTC Databricks MLOps
    dateFormat  YYYY-MM-DD
    section Risk Gate
    Validate network, UC, aliases, CI auth :active, 2026-06-01, 2d
    section Tuần 1: Setup & Ingestion
    Setup Git, Auth & UC Design      :active, 2026-06-03, 2d
    Ingestion direct or fallback     :active, 2026-06-05, 2d
    section Tuần 2: Feature & Model Pipeline
    Feature Engineering & Pytest    : 2026-06-08, 2d
    Optuna Training Notebook        : 2026-06-10, 3d
    section Tuần 3: Model Registry & CI/CD
    Champion/Challenger Aliases     : 2026-06-15, 2d
    Cấu hình DABs dev/prod targets  : 2026-06-17, 3d
    section Tuần 4: Deploy & Monitor
    Deploy Workflows & Jobs         : 2026-06-22, 2d
    AI/BI Dashboard & SQL Alerts    : 2026-06-24, 3d
```

### Giảm thiểu Rủi ro:
1. **Outbound Block:** Nạp dữ liệu gián tiếp qua GitHub Actions Runner hoặc AWS Lambda -> UC Volume.
2. **Cạn kiệt Compute Quota:** Bắt đầu với 15 trials, dùng pruner dừng sớm, đặt timeout và giãn retrain lên 12h/lần nếu cần.
3. **CI/CD Token Risk:** Dùng PAT ngắn hạn, protected environments, rotation và quyền tối thiểu.
4. **DAB multi-target chưa sẵn sàng:** Dùng một target/dev trước, prod deploy thủ công hoặc deferred đến Tuần 3.
5. **Monitoring feature bị giới hạn:** Deferred A6; khi triển khai thì dùng Delta metrics tables và dashboard thủ công nếu managed features không đủ.
6. **Optuna quota chưa xác minh:** Deferred A7; bắt đầu bằng baseline model, sau đó mới bật tuning với trials thấp.
7. **Trôi lệch dữ liệu (Drift):** Định kỳ ghi nhận thực tế (`actual_close`) để tính toán sai số (RMSE/MAPE) của `@Champion` trên dashboard.

---

## 5. Hành động Tiếp theo (Next Steps)

1. **Viết GitHub Actions ingestion trung gian:** Dùng `python-binance` lấy BTC hourly candles, lưu CSV/Parquet artifact hoặc file tạm, rồi upload vào UC Volume.
2. **Thiết kế Databricks load/merge job trên UC:** Đọc file từ UC Volume đã xác minh hoạt động, validate schema, deduplicate theo timestamp và merge vào Delta raw table.
3. **Triển khai Champion/Challenger bằng aliases:** Dùng `@Challenger` cho model mới và promote sang `@Champion` khi metric tốt hơn.
4. **Cấu hình CI/CD token:** Lưu `DATABRICKS_HOST` và `DATABRICKS_TOKEN` trong GitHub Secrets/Environments, chạy validate/deploy tối thiểu cho dev trước.
5. **Training baseline trước Optuna:** Xây dựng training pipeline tối thiểu trước, chưa cần chạy Optuna quota spike ngay.
6. **Deferred A5 sang Tuần 3:** Chưa cần hoàn thiện DAB targets `dev`/`prod` trong Tuần 1-2.
7. **Deferred A6/A7:** Monitoring/alerts và Optuna quota validation xử lý sau khi pipeline data + baseline training chạy được.

---

## 6. Kết luận

Kế hoạch triển khai BTC Databricks MLOps khả thi ở mức demo/end-to-end nếu chủ động giảm rủi ro Free Edition. Network spike đã loại bỏ phương án ingestion trực tiếp từ Databricks; ingestion nên chạy qua runner trung gian dùng `python-binance`, sau đó upload dữ liệu vào UC Volume để Databricks xử lý tiếp. Spike A2 đã xác nhận UC Volume/Table hoạt động, nên landing/staging và Delta raw layer có thể triển khai theo thiết kế hiện tại. Spike A3 đã xác nhận MLflow Model Aliases hoạt động, nên Champion/Challenger có thể triển khai bằng `@Champion` và `@Challenger` như thiết kế. CI/CD sẽ dùng Databricks token/PAT, cần giảm rủi ro bằng token ngắn hạn, rotation và GitHub protected environments. DAB multi-target `dev`/`prod` tạm deferred sang Tuần 3, nên Tuần 1-2 có thể dùng một target/dev hoặc deploy thủ công tối thiểu để không chặn pipeline. A6 monitoring/alerts và A7 Optuna quota validation cũng deferred; trước mắt nên tập trung hoàn thiện ingestion qua runner trung gian, UC load/merge, feature engineering và baseline training. Sau khi pipeline cơ bản chạy được, mới quay lại xử lý monitoring và Optuna tuning/quota.
