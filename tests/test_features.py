import pytest
import pandas as pd

pyspark = pytest.importorskip("pyspark.sql")
SparkSession = pyspark.SparkSession


@pytest.fixture(scope="session")
def spark():
    try:
        spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()
    except Exception as e:
        pytest.skip(f"local PySpark runtime unavailable: {e}")
    yield spark
    spark.stop()


def test_compute_features_basic(spark):
    from src.data.features import compute_features
    import datetime

    start = datetime.datetime(2025, 1, 1, 0, 0)
    data = [
        (
            start + datetime.timedelta(hours=t),
            float(39900 + t),
            float(40100 + t),
            float(39800 + t),
            float(40000 + t),
            float(100 + t),
        )
        for t in range(24)
    ]
    df = spark.createDataFrame(
        data, ["open_time", "open", "high", "low", "close", "volume"]
    )
    result = compute_features(df, ma_windows=[7], lag_hours=[1])
    assert "ma_7" in result.columns
    assert "close_lag_1h" in result.columns
    assert "return_1h" in result.columns
    assert "hl_spread" in result.columns
    assert "oc_change" in result.columns
    assert "target_close_1h" in result.columns
    result_pd = result.toPandas()
    assert len(result_pd) == 24
    assert result_pd.loc[1, "close_lag_1h"] == 40000.0
    assert result_pd.loc[1, "return_1h"] == pytest.approx(1 / 40000.0)
    assert result_pd.loc[0, "target_close_1h"] == 40001.0


def test_compute_features_target_requires_exact_next_hour(spark):
    from src.data.features import compute_features
    import datetime

    start = datetime.datetime(2025, 1, 1, 0, 0)
    data = [
        (start, 39900.0, 40100.0, 39800.0, 40000.0, 100.0),
        (start + datetime.timedelta(hours=2), 39902.0, 40102.0, 39802.0, 40002.0, 102.0),
    ]
    df = spark.createDataFrame(
        data, ["open_time", "open", "high", "low", "close", "volume"]
    )

    result_pd = compute_features(df, ma_windows=[1], lag_hours=[1]).orderBy(
        "open_time"
    ).toPandas()

    assert pd.isna(result_pd.loc[0, "target_close_1h"])
