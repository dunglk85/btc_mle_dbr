import csv
import io

from src.data import binance_landing


def test_rows_to_csv_bytes_adds_source_and_header():
    rows = [
        {
            "open_time": "2026-06-01T00:00:00+00:00",
            "open": 100000.0,
            "high": 101000.0,
            "low": 99000.0,
            "close": 100500.0,
            "volume": 12.34,
            "close_time": "2026-06-01T00:59:59+00:00",
            "quote_volume": 1234567.89,
            "trades": 1000,
        }
    ]

    content = binance_landing.rows_to_csv_bytes(rows).decode("utf-8")
    parsed = list(csv.DictReader(io.StringIO(content)))

    assert parsed[0]["open_time"] == "2026-06-01T00:00:00+00:00"
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
    monkeypatch.setattr(binance_landing, "fetch_klines", lambda **_kwargs: raw)
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
