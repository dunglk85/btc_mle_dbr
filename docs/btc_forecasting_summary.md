# Tóm tắt trao đổi: Dự án dự đoán BTC hourly

## 1. Có thể dùng Hidden Markov Model (HMM) không?

Có thể dùng **HMM**, nhưng không nên dùng HMM làm mô hình chính để dự đoán giá BTC giờ tiếp theo.

HMM phù hợp hơn để phát hiện **market regimes** — các trạng thái thị trường ẩn, ví dụ:

| Hidden State | Ý nghĩa |
|---|---|
| State 0 | Bull market |
| State 1 | Bear market |
| State 2 | Sideway / low volatility |
| State 3 | High volatility |

Ví dụ cách dùng:

```python
X = df[["return_1h", "volatility_24h", "volume_zscore"]]

model = GaussianHMM(
    n_components=4,
    covariance_type="full",
    n_iter=100
)

model.fit(X)
df["market_state"] = model.predict(X)
```

Cách dùng hợp lý:

```text
OHLCV
  → Feature Engineering
  → HMM
  → market_state
  → XGBoost / LightGBM
  → Predict next hour
```

Kết luận: **HMM nên dùng như feature engineering / regime detection**, không nên dùng làm model dự đoán chính.

---

## 2. Có nên dùng LSTM không?

Có thể dùng **LSTM**, nhưng nên coi là **challenger nâng cao**, không phải baseline chính.

Thứ tự hợp lý:

```text
Baseline 1: Persistence model
Baseline 2: Linear Regression / Ridge
Challenger 1: XGBoost / LightGBM
Challenger 2: LSTM
Optional: HMM regime feature + XGBoost/LSTM
```

LSTM phù hợp khi muốn model học từ chuỗi nhiều giờ trước đó:

```text
Input: 24 giờ hoặc 48 giờ gần nhất
Output: return hoặc giá giờ tiếp theo
```

Ví dụ:

```text
X[t] = dữ liệu từ t-24 đến t
y[t] = return từ t đến t+1
```

Lưu ý khi dùng LSTM:

- Dữ liệu BTC hourly từ 2025 đến nay chỉ khoảng hơn 1 năm, không quá lớn cho deep learning.
- Dễ bị overfit.
- Cần scaling dữ liệu.
- Không được split random, phải split theo thời gian.

Ví dụ split:

```text
Train: 2025-01 → 2026-03
Validation: 2026-04
Test: 2026-05 → nay
```

Kết luận: **XGBoost/LightGBM nên là challenger chính; LSTM là phần mở rộng nâng cao.**

---

## 3. Áp dụng phân tích kỹ thuật trong tài chính

Có thể áp dụng các phương pháp **Technical Analysis (TA)**, nhưng nên dùng chúng làm **feature cho mô hình ML**, không dùng như rule trading cứng.

Ví dụ không nên hiểu đơn giản:

```text
RSI > 70 → SELL
RSI < 30 → BUY
```

Mà nên dùng:

```text
RSI, MACD, ATR, Bollinger Band Width, volume_ratio...
```

làm input cho model.

### Nhóm feature nên dùng

#### Return features

```python
df["return_1h"] = df["close"] / df["close"].shift(1) - 1
df["return_6h"] = df["close"] / df["close"].shift(6) - 1
df["return_24h"] = df["close"] / df["close"].shift(24) - 1
```

#### Moving Average

```python
df["ma_7"] = df["close"].rolling(7).mean()
df["ma_24"] = df["close"].rolling(24).mean()
df["ma_168"] = df["close"].rolling(168).mean()
```

Nên chuyển sang ratio:

```python
df["close_ma7_ratio"] = df["close"] / df["ma_7"]
df["close_ma24_ratio"] = df["close"] / df["ma_24"]
df["close_ma168_ratio"] = df["close"] / df["ma_168"]
```

#### MACD

```text
MACD = EMA12 - EMA26
Signal = EMA9(MACD)
Histogram = MACD - Signal
```

Feature:

```text
macd
macd_signal
macd_hist
```

#### RSI

Feature phổ biến:

```text
rsi_14
```

Ý nghĩa trader thường dùng:

```text
RSI > 70: overbought
RSI < 30: oversold
```

Trong ML, dùng trực tiếp `rsi_14` làm feature.

#### Volatility

```text
volatility_24h
atr_14
atr_ratio = atr_14 / close
bb_width
```

#### Volume

```python
df["volume_ma24"] = df["volume"].rolling(24).mean()
df["volume_ratio"] = df["volume"] / df["volume_ma24"]
```

Có thể dùng thêm:

```python
df["log_volume"] = np.log1p(df["volume"])
```

#### Time features

Crypto giao dịch 24/7 nhưng vẫn có chu kỳ theo phiên giao dịch.

Feature nên dùng:

```text
hour_sin
hour_cos
weekday_sin
weekday_cos
```

---

## 4. `hour_sin` và `hour_cos` là gì?

`hour_sin` và `hour_cos` là cách mã hóa giờ theo dạng tuần hoàn.

Nếu dùng trực tiếp:

```text
23, 0, 1
```

model có thể hiểu sai rằng 23 cách 0 rất xa, trong khi thực tế 23:00 và 00:00 chỉ cách nhau 1 giờ.

Công thức:

```python
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
```

Ví dụ:

| Hour | hour_sin | hour_cos |
|---|---:|---:|
| 0 | 0.000 | 1.000 |
| 6 | 1.000 | 0.000 |
| 12 | 0.000 | -1.000 |
| 18 | -1.000 | 0.000 |
| 23 | -0.259 | 0.966 |

Tương tự cho ngày trong tuần:

```python
df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)
```

Kết luận: với XGBoost/LightGBM, nên tạo cả:

```text
hour
hour_sin
hour_cos
weekday
weekday_sin
weekday_cos
```

rồi để model tự chọn feature quan trọng.

---

## 5. Có nên normalization các feature về giá không?

Phụ thuộc vào model.

### Với XGBoost / LightGBM / Random Forest

Thường **không cần normalization**.

Tree-based models không quá nhạy với scale vì chúng split theo ngưỡng:

```text
close > 100000?
rsi > 70?
volume > 2e9?
```

Do đó scale feature không quan trọng bằng việc tạo feature đúng.

### Với LSTM / Neural Network

Nên normalization hoặc standardization.

Ví dụ:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)
```

Quan trọng: chỉ fit scaler trên train, sau đó transform validation/test để tránh data leakage.

---

## 6. Với feature giá, nên dùng giá tuyệt đối hay relative features?

Với dữ liệu tài chính, nên ưu tiên:

```text
return
ratio
volatility-normalized features
```

thay vì dùng trực tiếp:

```text
close
open
high
low
volume
```

Vì giá BTC thay đổi rất mạnh theo thời gian. Ví dụ BTC ở 30k và BTC ở 120k có scale rất khác nhau.

Nên dùng:

```python
df["return_1h"] = df["close"] / df["close"].shift(1) - 1
df["close_ma24_ratio"] = df["close"] / df["ma_24"]
df["atr_ratio"] = df["atr_14"] / df["close"]
df["volume_ratio"] = df["volume"] / df["volume_ma24"]
```

Kết luận: với XGBoost/LightGBM, không nhất thiết phải scale, nhưng nên **biến đổi feature giá thành return/ratio** để mô hình học ổn định hơn.

---

## 7. Target nên là `close` hay `return_1h`?

Dù dùng XGBoost/LightGBM/Random Forest, việc chọn target vẫn rất quan trọng.

### Target dạng close

```python
df["target_close"] = df["close"].shift(-1)
```

Model học:

```text
BTC hiện tại = 105000
→ BTC giờ sau = 105120
```

Vấn đề: model có thể đạt RMSE tốt chỉ bằng cách dự đoán gần giống giá hiện tại:

```text
close[t+1] ≈ close[t]
```

Nhìn metric có vẻ tốt nhưng có thể không hữu ích cho trading.

### Target dạng return

Nên dùng:

```python
df["target_return_1h"] = df["close"].shift(-1) / df["close"] - 1
```

Model học:

```text
Với trạng thái thị trường hiện tại, return của giờ tiếp theo là bao nhiêu?
```

Return thường ổn định hơn giá tuyệt đối.

### Log return

Có thể dùng:

```python
df["target_log_return_1h"] = np.log(df["close"].shift(-1) / df["close"])
```

Đây là target rất phổ biến trong quantitative finance.

---

## 8. Giải thích `close.pct_change().shift(-1)`

Dòng code:

```python
target = close.pct_change().shift(-1)
```

có nghĩa là lấy return của giờ tiếp theo làm target cho dòng hiện tại.

### Bước 1: `pct_change()`

```python
close.pct_change()
```

Tính:

```text
return_t = close_t / close_{t-1} - 1
```

Ví dụ:

| Time | Close | pct_change |
|---|---:|---:|
| t0 | 100 | NaN |
| t1 | 105 | 0.0500 |
| t2 | 102 | -0.0286 |
| t3 | 108 | 0.0588 |

### Bước 2: `shift(-1)`

```python
close.pct_change().shift(-1)
```

Dịch return lên 1 dòng:

| Time | Target |
|---|---:|
| t0 | 0.0500 |
| t1 | -0.0286 |
| t2 | 0.0588 |
| t3 | NaN |

Nghĩa là tại dòng `t0`, target là return từ `t0` đến `t1`.

Cách viết rõ hơn:

```python
df["target_return_1h"] = df["close"].shift(-1) / df["close"] - 1
```

Cách này dễ đọc hơn khi review code.

---

## 9. Khuyến nghị pipeline cho dự án BTC ở VSF

Pipeline đề xuất:

```text
Raw OHLCV
  → Feature Engineering
      - return features
      - lag features
      - rolling statistics
      - technical indicators
      - cyclical time features
      - optional HMM market_state
  → Train/Validation/Test split theo thời gian
  → Baseline models
  → XGBoost / LightGBM challenger
  → Optional LSTM challenger
  → Evaluate
  → Champion/Challenger comparison
```

### Feature set nên ưu tiên

```text
return_1h
return_6h
return_24h

ma_7
ma_24
ma_168

close_ma7_ratio
close_ma24_ratio
close_ma168_ratio

ema_12
ema_26

macd
macd_signal
macd_hist

rsi_14

atr_14
atr_ratio

bb_width

volume_ratio
log_volume

volatility_24h

hour_sin
hour_cos
weekday_sin
weekday_cos

market_state  # optional, từ HMM
```

### Target nên dùng

Ưu tiên:

```python
df["target_return_1h"] = df["close"].shift(-1) / df["close"] - 1
```

hoặc:

```python
df["target_log_return_1h"] = np.log(df["close"].shift(-1) / df["close"])
```

### Metric nên theo dõi

Với regression:

```text
RMSE
MAE
Directional Accuracy
```

Với classification up/down:

```text
Accuracy
Precision
Recall
F1
Directional Accuracy
```

Kết luận cuối:

```text
Model chính nên là LightGBM/XGBoost với target là next-hour return hoặc log return.
HMM dùng để tạo market_state.
LSTM dùng như challenger nâng cao.
Feature giá nên chuyển thành return/ratio thay vì dùng toàn bộ giá tuyệt đối.
```
