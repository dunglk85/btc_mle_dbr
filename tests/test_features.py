import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    spark = SparkSession.builder.master("local[1]").appName("test").getOrCreate()
    yield spark
    spark.stop()


def test_compute_features_basic(spark):
    from src.data.features import compute_features
    import datetime

    start = datetime.datetime(2025, 1, 1, 0, 0)
    data = [
        (start + datetime.timedelta(hours=t), float(40000 + t))
        for t in range(24)
    ]
    df = spark.createDataFrame(data, ["open_time", "close"])
    result = compute_features(df, ma_windows=[7], lag_hours=[1])
    assert result.columns
    result_pd = result.toPandas()
    assert len(result_pd) == 24
