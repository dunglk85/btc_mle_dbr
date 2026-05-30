import mlflow
import pandas as pd
from typing import Any
from sklearn.metrics import mean_squared_error


def track_performance(
    model: Any,
    X: pd.DataFrame,
    y_actual: pd.Series,
    y_pred: pd.Series | None = None,
    run_id: str | None = None,
) -> dict[str, float]:
    if y_pred is None:
        y_pred = pd.Series(model.predict(X))
    rmse = mean_squared_error(y_actual, y_pred) ** 0.5
    mape = (abs(y_actual - y_pred) / y_actual.abs()).mean()
    metrics = {"rmse": float(rmse), "mape": float(mape)}
    if run_id:
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metrics(metrics)
    return metrics
