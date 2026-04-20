#!/usr/bin/env python3
"""
Build local JSON cache for col_minors_dashboard.html.

Usage:
    python3 build_col_minors_cache.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests

BASE = "https://www.fangraphs.com/api/leaders/minor-league/data"
COMMON = (
    "pos=all&level=0&lg=2,4,5,6,7,8,9,10,11,14,12,13,15,16,17,18,30,32"
    "&qual=n&season=2026&seasonEnd=2026&org=19&ind=0&splitTeam=false&players="
)
OUT = Path("col_minors_cache.json")


def _request_with_clients(url: str) -> dict:
    """
    Try plain requests first, then cloudscraper for Cloudflare-protected responses.
    """
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.fangraphs.com/",
        "Origin": "https://www.fangraphs.com",
    }

    # Attempt 1: requests session (fast path)
    try:
        r = requests.get(url, timeout=30, headers=headers)
        if r.status_code < 400:
            return r.json()
    except Exception:
        pass

    # Attempt 2: cloudscraper fallback for Cloudflare 403s
    try:
        import cloudscraper  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "FanGraphs returned 403 and cloudscraper is not installed. "
            "Run: pip3 install cloudscraper"
        ) from exc

    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "darwin", "mobile": False}
    )

    last_err: Exception | None = None
    for i in range(3):
        try:
            resp = scraper.get(url, timeout=40, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.2 * (i + 1))

    raise RuntimeError(f"Failed to fetch after cloudscraper retries: {last_err}")


def fetch(stats: str, type_: int) -> dict:
    url = f"{BASE}?{COMMON}&stats={stats}&type={type_}"
    return _request_with_clients(url)


def main() -> None:
    print("Fetching FanGraphs payloads...")
    payloads = {
        "pa1": fetch("pit", 1),
        "pa2": fetch("pit", 2),
        "pa4": fetch("pit", 4),
        "ba1": fetch("bat", 1),
        "ba2": fetch("bat", 2),
        "ba4": fetch("bat", 4),
    }
    OUT.write_text(json.dumps({"payloads": payloads}, separators=(",", ":")))
    print(f"Wrote {OUT.resolve()}")


if __name__ == "__main__":
    main()
