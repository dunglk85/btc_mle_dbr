# Catalog Schema

## Multi-Environment Catalogs

| Catalog    | Purpose      |
|------------|-------------|
| `btc_dev`  | Development  |
| `btc_stg`  | Staging      |
| `btc_prod` | Production   |

## Schema Structure (per catalog)

```
<catalog>/
├── raw/
│   └── btc_hourly        # Raw OHLCV data from Binance
├── features/
│   └── btc_features      # Engineered features
└── predictions/
    └── btc_predictions   # Model predictions
```

## btc_hourly Schema

| Column        | Type      | Description                |
|---------------|-----------|----------------------------|
| open_time     | timestamp | Start of candle            |
| open          | double    | Open price                 |
| high          | double    | High price                 |
| low           | double    | Low price                  |
| close         | double    | Close price (target)       |
| volume        | double    | Volume                     |
| close_time    | timestamp | End of candle              |
| quote_volume  | double    | Quote asset volume         |
| trades        | bigint    | Number of trades           |
