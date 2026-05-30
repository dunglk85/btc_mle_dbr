import pytest
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
