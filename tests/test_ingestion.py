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


def test_fetch_klines_uses_python_binance_client(monkeypatch):
    client = FakeBinanceClient()

    monkeypatch.setattr(ingestion, "get_binance_client", lambda: client)

    rows = ingestion.fetch_klines(limit=1, start_time=12345)

    assert rows == client.rows
    assert client.calls[0]["startTime"] == 12345


def test_load_landing_to_raw_merges_volume_files(monkeypatch):
    monkeypatch.setattr(ingestion, "F", FakeFunctions())
    monkeypatch.setattr(ingestion, "Window", FakeWindow)
    spark = FakeLandingSpark()

    result = ingestion.load_landing_to_raw(
        spark,
        catalog="btc_dev",
        raw_schema="raw",
        volume_name="landing",
        table="btc_hourly",
    )

    assert spark.readStream.loaded_path == "/Volumes/btc_dev/raw/landing/btc_hourly"
    assert spark.readStream.format_name == "cloudFiles"
    assert spark.readStream.options["cloudFiles.format"] == "csv"
    assert spark.readStream.options["cloudFiles.schemaLocation"] == (
        "/Volumes/btc_dev/raw/landing/_schemas/btc_hourly"
    )
    assert spark.write_stream.options["checkpointLocation"] == (
        "/Volumes/btc_dev/raw/landing/_checkpoints/btc_hourly"
    )
    assert spark.write_stream.output_mode == "append"
    assert spark.write_stream.trigger_kwargs == {"availableNow": True}
    assert spark.write_stream.target_table == "btc_dev.raw.btc_hourly_landing_autoloader"
    assert any("CREATE VOLUME IF NOT EXISTS btc_dev.raw.landing" in sql for sql in spark.sql_calls)
    assert any(
        "CREATE TABLE IF NOT EXISTS btc_dev.raw.btc_hourly_landing_autoloader" in sql
        for sql in spark.sql_calls
    )
    assert any("MERGE INTO btc_dev.raw.btc_hourly" in sql for sql in spark.sql_calls)
    assert spark.temp_view == "_btc_hourly_landing"
    assert result == "table:btc_dev.raw.btc_hourly"


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


class FakeLandingSpark:
    def __init__(self):
        self.sql_calls = []
        self.readStream = FakeStreamReader(self)
        self.write_stream = None
        self.temp_view = None

    def sql(self, query):
        self.sql_calls.append(query)

    def table(self, table_name):
        if table_name.endswith("_landing_autoloader"):
            return FakeLandingDataFrame(self)
        return FakeTable(table_name)


class FakeStreamReader:
    def __init__(self, spark):
        self.spark = spark
        self.format_name = None
        self.options = {}
        self.loaded_path = None

    def format(self, name):
        self.format_name = name
        return self

    def option(self, key, value):
        self.options[key] = value
        return self

    def schema(self, _schema):
        return self

    def load(self, path):
        self.loaded_path = path
        return FakeLandingDataFrame(self.spark)


class FakeLandingDataFrame:
    def __init__(self, spark):
        self.spark = spark
        self.sparkSession = spark
        self.writeStream = FakeWriteStream(spark, self)

    def withColumn(self, _name, _value):
        return self

    def select(self, *_cols):
        return self

    def drop(self, *_cols):
        return self

    def filter(self, condition):
        if "IS NULL" in str(condition):
            return FakeCountDataFrame(0)
        return self

    def count(self):
        return 1

    def createOrReplaceTempView(self, name):
        self.spark.temp_view = name


class FakeWriteStream:
    def __init__(self, spark, batch_df):
        self.spark = spark
        self.batch_df = batch_df
        self.options = {}
        self.output_mode = None
        self.trigger_kwargs = None
        self.target_table = None

    def option(self, key, value):
        self.options[key] = value
        return self

    def outputMode(self, mode):
        self.output_mode = mode
        return self

    def trigger(self, **kwargs):
        self.trigger_kwargs = kwargs
        return self

    def toTable(self, table_name):
        self.spark.write_stream = self
        self.target_table = table_name
        return FakeQuery()


class FakeQuery:
    def awaitTermination(self):
        return None


class FakeFunctions:
    def coalesce(self, *_args):
        return FakeExpression()

    def col(self, name):
        return FakeColumn(name)

    def lit(self, value):
        return FakeExpression(value)

    def current_timestamp(self):
        return FakeExpression("current_timestamp")

    def to_timestamp(self, value, _format=None):
        return value

    def row_number(self):
        return FakeRowNumber()


class FakeExpression:
    def __init__(self, value=None):
        self.value = value

    def alias(self, _name):
        return self


class FakeColumn:
    def __init__(self, name):
        self.name = name

    def isNull(self):
        return f"{self.name} IS NULL"

    def cast(self, _type):
        return self

    def alias(self, _name):
        return self

    def desc(self):
        return self

    def __eq__(self, other):
        return f"{self.name} = {other}"

    def __ge__(self, other):
        return f"{self.name} >= {other}"


class FakeWindow:
    @staticmethod
    def partitionBy(*_cols):
        return FakeWindowSpec()


class FakeWindowSpec:
    def orderBy(self, *_cols):
        return self


class FakeRowNumber:
    def over(self, _window):
        return self


class FakeCountDataFrame:
    def __init__(self, count):
        self._count = count

    def count(self):
        return self._count
