import os
import yaml
from typing import Any
from pathlib import Path


def load_config(env: str = "dev") -> dict[str, Any]:
    config = {
        "catalog": os.getenv("CATALOG", f"btc_{env}"),
        "raw_schema": "raw",
        "features_schema": "features",
        "predictions_schema": "predictions",
        "landing_volume": "landing",
        "binance_symbol": "BTCUSDT",
        "interval": "1h",
        "retrain_interval_hours": int(os.getenv("RETRAIN_INTERVAL", "3")),
        "optuna_n_trials": int(os.getenv("OPTUNA_TRIALS", "50")),
        "model_type": os.getenv("MODEL_TYPE", "xgboost"),
    }
    override_path = Path(f"configs/{env}.yaml")
    if override_path.exists():
        with open(override_path) as f:
            config.update(yaml.safe_load(f))
    return config
