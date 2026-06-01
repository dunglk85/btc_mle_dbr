from __future__ import annotations

import argparse
import csv
import io
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Sequence

import requests

from src.data.ingestion import klines_to_rows


DEFAULT_VOLUME_PATH = "/Volumes/btc_dev/raw/landing/btc_hourly"
BINANCE_VISION_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"


@dataclass(frozen=True)
class LandingUploadResult:
    candle_count: int
    file_path: str


def rows_to_csv_bytes(rows: Iterable[dict]) -> bytes:
    fieldnames = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
        "source",
    ]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        normalized = dict(row)
        normalized["open_time"] = _format_timestamp(normalized["open_time"])
        normalized["close_time"] = _format_timestamp(normalized["close_time"])
        normalized.setdefault("source", "binance")
        writer.writerow({name: normalized.get(name) for name in fieldnames})
    return output.getvalue().encode("utf-8")


def upload_bytes_to_volume(
    client: Any,
    contents: bytes,
    file_path: str,
    overwrite: bool = True,
) -> None:
    client.files.upload(
        file_path=file_path,
        contents=io.BytesIO(contents),
        overwrite=overwrite,
    )


def fetch_and_upload_landing_file(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 24,
    volume_path: str = DEFAULT_VOLUME_PATH,
    start_time: int | None = None,
    end_time: int | None = None,
    client: Any | None = None,
) -> LandingUploadResult:
    raw = fetch_klines_from_binance_vision(
        symbol=symbol,
        interval=interval,
        limit=limit,
        start_time=start_time,
        end_time=end_time,
    )
    rows = klines_to_rows(raw)
    for row in rows:
        row["source"] = "binance"

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_path = f"{volume_path.rstrip('/')}/btc_hourly_{timestamp}.csv"
    upload_bytes_to_volume(client or _workspace_client(), rows_to_csv_bytes(rows), file_path)
    return LandingUploadResult(candle_count=len(rows), file_path=file_path)


def fetch_klines_from_binance_vision(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 24,
    start_time: int | None = None,
    end_time: int | None = None,
) -> list:
    params: dict[str, int | str] = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    if start_time is not None:
        params["startTime"] = start_time
    if end_time is not None:
        params["endTime"] = end_time

    response = requests.get(BINANCE_VISION_KLINES_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch Binance BTC hourly candles and upload CSV to UC Volume."
    )
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--volume-path", default=DEFAULT_VOLUME_PATH)
    parser.add_argument("--start-time", type=int, default=None)
    parser.add_argument("--end-time", type=int, default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)
    result = fetch_and_upload_landing_file(
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
        volume_path=args.volume_path,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    print(f"Fetched {result.candle_count} candles")
    print(f"Uploaded {result.file_path}")
    return 0


def _format_timestamp(value: object) -> object:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return value


def _workspace_client() -> Any:
    try:
        from databricks.sdk import WorkspaceClient
    except ImportError as e:
        raise RuntimeError(
            "databricks-sdk is required for UC Volume uploads. "
            "Install it with `pip install databricks-sdk`."
        ) from e
    return WorkspaceClient()


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()
