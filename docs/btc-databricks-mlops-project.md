Nhiệm vụ:

Dữ liệu: 
Kéo dữ liệu về; vì hourly, nên anh có thể kéo dữ liệu từ 20250101 -> nay (hơn 1 năm) 
Xử lí và lưu trữ theo dạng delta table trên catalog của databricks. Có 2 layer cần lưu trên Catalog: raw -> features
Cần hiểu được sự khác biệt của Catalog vs DBFS, delta vs parquet
Training model: Cơ chế champion vs challenger
Train model: cơ bản, dùng các thuật toán nào cũng được, có grid search, có tự chọn được best model --> Challenger model
Compare model: Challenger vs Champion (best model hiện tại trên prod) -> Model nào thắng thì sẽ thực hiện predict cho điểm dữ liệu tiếp theo (next 1 hour)
Cần hiểu được dữ liệu ở đây là timeseries, nên không random split được như thông thường
Sử dụng git actions để thiết lập CICD và chạy job hàng giờ
Thiết lập monitoring: 
Data quality
Model performance
Job quality
Mình đã gửi tài liệu kiến trúc ML Platform (Tổng quát) 

Tuần 1: Thiết kế kiến trúc, và nghiên cứu tính năng trên Databricks -> Anh sẽ cần biết tính năng nào tương ứng trên Databricks để thực hiện được các task trên. 

Các tuần sau thì mình triển khai mấy nhiệm vụ trên

Gợi ý 1 số tính năng/keywords có thể dùng trên databricks: 

Catalog, MLflow, feature registry, Data quality, Dashboard
Gần như tất cả các tính năng có thể tạo/quản lý được bằng terminal ở local HOẶC kéo thả trên UI