import pandas as pd
import numpy as np


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


def test_temporal_split_preserves_time_order():
    from src.models.training import temporal_train_test_split

    df = pd.DataFrame(
        {
            "open_time": pd.date_range("2025-01-01", periods=10, freq="h"),
            "feature": range(10),
            "close": range(100, 110),
        }
    )

    X_train, y_train, X_test, y_test = temporal_train_test_split(df, test_fraction=0.2)

    assert list(X_train["feature"]) == list(range(8))
    assert list(X_test["feature"]) == [8, 9]
    assert list(y_train) == list(range(100, 108))
    assert list(y_test) == [108, 109]


def test_train_baseline_random_forest_returns_metrics():
    from src.models.training import train_baseline_random_forest

    X = pd.DataFrame({"feature": np.arange(120)})
    y = pd.Series(np.arange(120, dtype=float))

    model, metrics = train_baseline_random_forest(
        X.iloc[:100], y.iloc[:100], X.iloc[100:], y.iloc[100:]
    )

    assert model is not None
    assert "rmse" in metrics
    assert "mae" in metrics
