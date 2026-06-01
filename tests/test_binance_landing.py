import csv
import io

from src.data import binance_landing


def test_rows_to_csv_bytes_adds_source_and_header():
    rows = [
        {
            "open_time": "2026-06-01 00:00:00",
            "open": 100000.0,
            "high": 101000.0,
            "low": 99000.0,
            "close": 100500.0,
            "volume": 12.34,
            "close_time": "2026-06-01 00:59:59",
            "quote_volume": 1234567.89,
            "trades": 1000,
        }
    ]

    content = binance_landing.rows_to_csv_bytes(rows).decode("utf-8")
    parsed = list(csv.DictReader(io.StringIO(content)))

    assert parsed[0]["open_time"] == "2026-06-01 00:00:00"
    assert parsed[0]["source"] == "binance"
    assert parsed[0]["trades"] == "1000"


def test_fetch_and_upload_landing_file_uploads_to_volume(monkeypatch):
    raw = [
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
    monkeypatch.setattr(
        binance_landing, "fetch_klines_from_binance_vision", lambda **_kwargs: raw
    )
    client = FakeWorkspaceClient()

    result = binance_landing.fetch_and_upload_landing_file(
        limit=1,
        volume_path="/Volumes/btc_dev/raw/landing/btc_hourly",
        client=client,
    )

    assert result.candle_count == 1
    assert result.file_path.startswith("/Volumes/btc_dev/raw/landing/btc_hourly/btc_hourly_")
    assert client.files.uploads[0]["file_path"] == result.file_path
    assert b"open_time,open,high,low,close" in client.files.uploads[0]["contents"]


def test_fetch_klines_from_binance_vision_uses_data_api(monkeypatch):
    calls = {}

    def fake_get(url, params, timeout):
        calls["url"] = url
        calls["params"] = params
        calls["timeout"] = timeout
        return FakeResponse([[1700000000000, "40000.0"]])

    monkeypatch.setattr(binance_landing.requests, "get", fake_get)

    result = binance_landing.fetch_klines_from_binance_vision(
        symbol="BTCUSDT",
        interval="1h",
        limit=1,
        start_time=123,
        end_time=456,
    )

    assert calls["url"] == "https://data-api.binance.vision/api/v3/klines"
    assert calls["params"] == {
        "symbol": "BTCUSDT",
        "interval": "1h",
        "limit": 1,
        "startTime": 123,
        "endTime": 456,
    }
    assert calls["timeout"] == 30
    assert result == [[1700000000000, "40000.0"]]


def test_fetch_klines_from_binance_vision_paginates(monkeypatch):
    calls = []
    first_page = [[1000, "1"] for _ in range(1000)]
    first_page[-1][0] = 1999
    second_page = [[2000, "2"], [3000, "3"]]

    def fake_get(_url, params, timeout):
        assert timeout == 30
        calls.append(params)
        payload = first_page if len(calls) == 1 else second_page
        return FakeResponse(payload)

    monkeypatch.setattr(binance_landing.requests, "get", fake_get)

    result = binance_landing.fetch_klines_from_binance_vision(limit=1002, start_time=1000)

    assert len(result) == 1002
    assert calls[0]["limit"] == 1000
    assert calls[0]["startTime"] == 1000
    assert calls[1]["limit"] == 2
    assert calls[1]["startTime"] == 2000


def test_fetch_and_upload_landing_file_accepts_start_date(monkeypatch):
    calls = {}
    raw = [
        [
            1735689600000,
            "40000.0",
            "41000.0",
            "39000.0",
            "40500.0",
            "100.0",
            1735693199999,
            "4000000.0",
            1000,
        ]
    ]

    def fake_fetch(**kwargs):
        calls.update(kwargs)
        return raw

    monkeypatch.setattr(binance_landing, "fetch_klines_from_binance_vision", fake_fetch)

    binance_landing.fetch_and_upload_landing_file(
        limit=1,
        start_date="2025-01-01",
        client=FakeWorkspaceClient(),
    )

    assert calls["start_time"] == 1735689600000


class FakeWorkspaceClient:
    def __init__(self):
        self.files = FakeFiles()


class FakeFiles:
    def __init__(self):
        self.uploads = []

    def upload(self, file_path, contents, overwrite):
        self.uploads.append(
            {
                "file_path": file_path,
                "contents": contents.read(),
                "overwrite": overwrite,
            }
        )


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload
