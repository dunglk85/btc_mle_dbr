from src.data import ingestion
from src.data.ingestion import klines_to_rows


def test_klines_to_rows():
    raw = [
        [1700000000000, "40000.0", "41000.0", "39000.0", "40500.0", "100.0", 1700003600000, "4000000.0", 1000]
    ]
    rows = klines_to_rows(raw)
    assert len(rows) == 1
    assert rows[0]["open"] == 40000.0
    assert rows[0]["close"] == 40500.0
    assert rows[0]["trades"] == 1000


def test_initial_ingest_uses_backfill_start_and_creates_table(monkeypatch):
    calls = {}

    def fake_fetch_klines(**kwargs):
        calls["fetch_kwargs"] = kwargs
        return [
            [
                1700000000000,
                "40000.0",
                "41000.0",
                "39000.0",
                "40500.0",
                "100.0",
                1700003600000,
                "4000000.0",
                1000,
            ]
        ]

    monkeypatch.setattr(ingestion, "fetch_klines", fake_fetch_klines)

    spark = FakeSpark(table_missing=True)

    result = ingestion.incremental_ingest(
        spark,
        catalog="btc_dev",
        schema="raw",
        table="btc_hourly",
        backfill_start_ms=12345,
    )

    assert calls["fetch_kwargs"]["start_time"] == 12345
    assert spark.saved_table == "btc_dev.raw.btc_hourly"
    assert result == "table:btc_dev.raw.btc_hourly"


def test_incremental_ingest_merges_existing_table(monkeypatch):
    calls = {}

    def fake_fetch_klines(**kwargs):
        calls["fetch_kwargs"] = kwargs
        return [
            [
                1700003600001,
                "40500.0",
                "41000.0",
                "40000.0",
                "40750.0",
                "90.0",
                1700007200000,
                "3650000.0",
                900,
            ]
        ]

    monkeypatch.setattr(ingestion, "fetch_klines", fake_fetch_klines)
    monkeypatch.setattr(
        ingestion, "get_latest_timestamp", lambda _spark, _table: 1700003600000
    )

    spark = FakeSpark(table_missing=False)

    ingestion.incremental_ingest(spark, catalog="btc_dev", schema="raw", table="btc_hourly")

    assert calls["fetch_kwargs"]["start_time"] == 1700003600001
    assert any("MERGE INTO btc_dev.raw.btc_hourly" in sql for sql in spark.sql_calls)


def test_fetch_klines_uses_python_binance_client_when_credentials_exist(monkeypatch):
    client = FakeBinanceClient()

    monkeypatch.setenv("BINANCE_API_KEY", "test-key")
    monkeypatch.setenv("BINANCE_API_SECRET", "test-secret")
    monkeypatch.setattr(ingestion, "get_binance_client", lambda: client)

    rows = ingestion.fetch_klines(limit=1, start_time=12345)

    assert rows == client.rows
    assert client.calls[0]["startTime"] == 12345


class FakeSpark:
    def __init__(self, table_missing):
        self.table_missing = table_missing
        self.sql_calls = []
        self.saved_table = None

    def sql(self, query):
        self.sql_calls.append(query)

    def table(self, table_name):
        if self.table_missing and not self.saved_table:
            raise Exception("missing table")
        return FakeTable(table_name)

    def createDataFrame(self, _rows, schema=None):
        return FakeDataFrame(self, schema)


class FakeTable:
    def __init__(self, table_name):
        self.table_name = table_name

    def limit(self, _count):
        return self

    def collect(self):
        return []

    def __eq__(self, other):
        return other == f"table:{self.table_name}"


class FakeDataFrame:
    def __init__(self, spark, _schema):
        self.spark = spark
        self.write = FakeWriter(spark)

    def createOrReplaceTempView(self, name):
        self.spark.temp_view = name


class FakeWriter:
    def __init__(self, spark):
        self.spark = spark

    def format(self, _format):
        return self

    def mode(self, _mode):
        return self

    def saveAsTable(self, table):
        self.spark.saved_table = table


class FakeBinanceClient:
    def __init__(self):
        self.calls = []
        self.rows = [
            [
                1700000000000,
                "40000.0",
                "41000.0",
                "39000.0",
                "40500.0",
                "100.0",
                1700003600000,
                "4000000.0",
                1000,
            ]
        ]

    def get_klines(self, **kwargs):
        self.calls.append(kwargs)
        return self.rows
