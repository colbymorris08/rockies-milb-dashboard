#!/usr/bin/env python3
"""
Build local JSON cache for col_minors_dashboard.html.

Usage:
    python3 build_col_minors_cache.py
    python3 build_col_minors_cache.py --headed --interactive   # if Cloudflare blocks headless

Why direct HTTP often fails now:
    FanGraphs sits behind Cloudflare (and similar checks). Those systems fingerprint TLS
    and JavaScript — plain ``requests`` / cloudscraper used to slip through but are
    commonly blocked with 403 today. This script tries HTTP first, then optionally
    ``curl_cffi`` (Chrome TLS impersonation), then launches Chromium via Playwright
    (real browser) so one command still refreshes the minors cache.

Environment:
    FG_SEASON          Season year (default 2026). CLI --season overrides.
    FG_COOKIE / FG_COOKIE_FILE   Optional browser Cookie string for HTTP path only.
    FG_NO_PLAYWRIGHT=1 Skip browser fallback (fail fast if HTTP blocked).
    FG_HTTP_ONLY=1    Same as --http-only

Optional extras:
    pip3 install curl_cffi          Often fixes 403 without a browser.
    pip3 install playwright && playwright install chromium   Required for browser fallback.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import requests

BASE = "https://www.fangraphs.com/api/leaders/minor-league/data"
LG = "2,4,5,6,7,8,9,10,11,14,12,13,15,16,17,18,30,32"
OUT = Path("col_minors_cache.json")


def _season(cli_season: int | None) -> int:
    if cli_season is not None:
        return cli_season
    return int(os.getenv("FG_SEASON", "2026"))


def _cookie_string() -> str:
    raw = os.getenv("FG_COOKIE", "").strip()
    path = os.getenv("FG_COOKIE_FILE", "").strip()
    if not raw and path:
        try:
            raw = Path(path).expanduser().read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return raw


def _headers() -> dict[str, str]:
    h: dict[str, str] = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.fangraphs.com/leaders/minor-league",
        "Origin": "https://www.fangraphs.com",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin",
    }
    ck = _cookie_string()
    if ck:
        h["Cookie"] = ck
    return h


def build_url(
    stats: str,
    type_: int,
    season: int,
    *,
    qual: str,
    include_lg: bool,
    include_org: bool,
) -> str:
    lg_part = f"&lg={LG}" if include_lg else "&lg="
    org_part = "&org=19" if include_org else ""
    return (
        f"{BASE}?pos=all&level=0{lg_part}&qual={qual}&season={season}&seasonEnd={season}"
        f"{org_part}&ind=0&splitTeam=false&players=&stats={stats}&type={type_}"
    )


def _request_json(url: str, headers: dict[str, str]) -> dict:
    r = requests.get(url, timeout=45, headers=headers)
    r.raise_for_status()
    return r.json()


def _request_with_clients(url: str, headers: dict[str, str]) -> dict:
    """requests → curl_cffi (optional) → cloudscraper."""
    try:
        return _request_json(url, headers)
    except Exception:
        pass

    try:
        from curl_cffi import requests as curl_requests  # type: ignore

        r = curl_requests.get(url, impersonate="chrome131", headers=headers, timeout=50)
        r.raise_for_status()
        return r.json()
    except Exception:
        pass

    try:
        import cloudscraper  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "FanGraphs blocked the request. Install: pip3 install cloudscraper curl_cffi\n"
            "Or rely on Playwright fallback / FG_COOKIE (see docstring)."
        ) from exc

    scraper = cloudscraper.create_scraper(
        browser={"browser": "chrome", "platform": "darwin", "mobile": False}
    )
    last_err: Exception | None = None
    for i in range(3):
        try:
            resp = scraper.get(url, timeout=50, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(1.2 * (i + 1))
    raise last_err or RuntimeError("unknown fetch error")


def fetch_one_http(stats: str, type_: int, season: int, headers: dict[str, str]) -> dict:
    candidates = [
        ("0", True, True),
        ("0", False, True),
        ("n", True, True),
        ("1", True, True),
        ("0", True, False),
    ]
    last_err: Exception | None = None
    for qual, include_lg, include_org in candidates:
        url = build_url(
            stats, type_, season, qual=qual, include_lg=include_lg, include_org=include_org
        )
        try:
            return _request_with_clients(url, headers)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            time.sleep(0.35)
    raise RuntimeError(
        f"FanGraphs blocked all HTTP variants for stats={stats!r} type={type_}: {last_err}"
    ) from last_err


def fetch_all_http(season: int) -> dict[str, dict]:
    headers = _headers()
    return {
        "pa1": fetch_one_http("pit", 1, season, headers),
        "pa2": fetch_one_http("pit", 2, season, headers),
        "pa4": fetch_one_http("pit", 4, season, headers),
        "ba1": fetch_one_http("bat", 1, season, headers),
        "ba2": fetch_one_http("bat", 2, season, headers),
        "ba4": fetch_one_http("bat", 4, season, headers),
    }


def _playwright_help() -> str:
    return (
        "Install browser automation:\n"
        "  pip3 install playwright\n"
        "  playwright install chromium\n"
        "Then re-run this script, or use:\n"
        "  python3 build_col_minors_cache_playwright.py --headed --interactive"
    )


def fetch_all_via_playwright(season: int, headed: bool, interactive: bool) -> dict[str, dict]:
    try:
        from build_col_minors_cache_playwright import fetch_minor_payloads_async
    except ImportError as exc:
        raise RuntimeError(_playwright_help()) from exc
    return asyncio.run(
        fetch_minor_payloads_async(season, headed=headed, interactive=interactive)
    )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Build col_minors_cache.json from FanGraphs.")
    p.add_argument("--season", type=int, default=None, help="Season (default: FG_SEASON or 2026)")
    p.add_argument(
        "--http-only",
        action="store_true",
        help="Do not use Playwright even if HTTP fails",
    )
    p.add_argument(
        "--headed",
        action="store_true",
        help="With browser fallback: show Chromium (helps when headless is blocked)",
    )
    p.add_argument(
        "--interactive",
        action="store_true",
        help="With browser fallback: pause for manual Cloudflare check",
    )
    p.add_argument(
        "--browser",
        action="store_true",
        help="Skip HTTP; fetch only via Playwright (faster when you know HTTP will 403)",
    )
    args = p.parse_args(argv)

    season = _season(args.season)
    http_only = args.http_only or os.getenv("FG_HTTP_ONLY", "").strip() in ("1", "true", "yes")
    no_pw = os.getenv("FG_NO_PLAYWRIGHT", "").strip() in ("1", "true", "yes")

    print("Fetching FanGraphs payloads...")
    if _cookie_string():
        print("  (using FG_COOKIE / FG_COOKIE_FILE for HTTP)")
    print(f"  season={season}")

    if args.browser:
        print("  (--browser: using Playwright only)")
        payloads = fetch_all_via_playwright(season, args.headed, args.interactive)
    else:
        try:
            payloads = fetch_all_http(season)
        except Exception as http_exc:
            if http_only or no_pw:
                print(
                    "\nFanGraphs is blocking scripted HTTP. Options:\n"
                    "  • Unset FG_NO_PLAYWRIGHT / omit --http-only so this script can use Playwright.\n"
                    "  • export FG_COOKIE='…' from your browser after visiting fangraphs.com.\n"
                    "  • pip3 install curl_cffi  (sometimes restores HTTP-only access)\n",
                    file=sys.stderr,
                )
                raise
            print(
                "\nDirect HTTP was blocked (403/Cloudflare is common now). "
                "Trying Chromium via Playwright...\n",
                file=sys.stderr,
            )
            try:
                payloads = fetch_all_via_playwright(season, args.headed, args.interactive)
            except Exception as pw_exc:
                raise RuntimeError(
                    f"HTTP failed: {http_exc!r}\nPlaywright failed: {pw_exc!r}\n\n{_playwright_help()}"
                ) from pw_exc

    OUT.write_text(json.dumps({"payloads": payloads}, separators=(",", ":")))
    print(f"Wrote {OUT.resolve()}")


if __name__ == "__main__":
    main()
