import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error


def test_evaluate_metrics():
    from src.models.evaluation import evaluate
    from sklearn.ensemble import RandomForestRegressor

    np.random.seed(42)
    X = pd.DataFrame({"a": np.random.rand(50), "b": np.random.rand(50)})
    y = pd.Series(2 * X["a"] + 3 * X["b"] + np.random.randn(50) * 0.1)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    metrics = evaluate(model, X, y)
    assert "rmse" in metrics
    assert "mae" in metrics
    assert metrics["rmse"] < 1.0
