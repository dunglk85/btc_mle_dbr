import optuna
import mlflow
import pandas as pd
from typing import Callable
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


SEARCH_SPACES: dict[str, Callable] = {
    "xgboost": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    },
    "lightgbm": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "num_leaves": trial.suggest_int("num_leaves", 8, 128),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
    },
    "random_forest": lambda trial: {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
    },
}


def optimize(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    model_type: str = "xgboost",
    n_trials: int = 50,
    experiment_name: str = "btc_model_tuning",
) -> optuna.Study:
    mlflow.set_experiment(experiment_name)

    def objective(trial):
        params = SEARCH_SPACES[model_type](trial)
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            model = _build_model(model_type, params)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            rmse = mean_squared_error(y_val, preds) ** 0.5
            mlflow.log_metric("rmse", rmse)
        return rmse

    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)
    return study


def _build_model(model_type: str, params: dict):
    if model_type == "xgboost":
        import xgboost
        return xgboost.XGBRegressor(**params, random_state=42)
    elif model_type == "lightgbm":
        import lightgbm
        return lightgbm.LGBMRegressor(**params, random_state=42, verbose=-1)
    elif model_type == "random_forest":
        return RandomForestRegressor(**params, random_state=42)
    raise ValueError(f"Unknown model type: {model_type}")


def temporal_train_test_split(
    df: pd.DataFrame,
    time_col: str = "open_time",
    target_col: str = "close",
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    sorted_df = df.sort_values(time_col).dropna().reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1 - test_fraction))
    feature_cols = [c for c in sorted_df.columns if c not in (time_col, target_col)]
    train = sorted_df.iloc[:split_idx]
    test = sorted_df.iloc[split_idx:]
    return train[feature_cols], train[target_col], test[feature_cols], test[target_col]


def train_baseline_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[RandomForestRegressor, dict[str, float]]:
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics = {
        "rmse": float(mean_squared_error(y_test, preds) ** 0.5),
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
        "mape": float((abs(y_test - preds) / y_test.abs()).mean()),
    }
    return model, metrics
