import mlflow
import pandas as pd
from typing import Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


METRICS = {
    "rmse": lambda y, p: mean_squared_error(y, p) ** 0.5,
    "mae": mean_absolute_error,
    "r2": r2_score,
    "mape": lambda y, p: (abs(y - p) / y.abs()).mean(),
}


def evaluate(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    prefix: str = "",
) -> dict[str, float]:
    preds = model.predict(X_test)
    results = {}
    for name, fn in METRICS.items():
        key = f"{prefix}{name}"
        results[key] = float(fn(y_test, preds))
    return results


def promote_if_better(
    challenger: Any,
    champion_uri: str | None,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[bool, dict[str, float]]:
    chal_metrics = evaluate(challenger, X_test, y_test, prefix="challenger_")
    if champion_uri is None:
        mlflow.pyfunc.log_model("model", python_model=challenger)
        return True, chal_metrics

    champion = mlflow.pyfunc.load_model(champion_uri)
    cham_metrics = evaluate(champion, X_test, y_test, prefix="champion_")
    combined = {**cham_metrics, **chal_metrics}
    return chal_metrics["challenger_rmse"] < cham_metrics["champion_rmse"], combined
