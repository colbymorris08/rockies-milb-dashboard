#!/usr/bin/env python3
"""
Automatically build col_minors_cache.json from FanGraphs using Playwright.

Why this exists:
  FanGraphs API often returns 403 to plain HTTP clients. A browser context can
  pass Cloudflare checks and then fetch the same API payloads.

Usage:
  python3 build_col_minors_cache_playwright.py
  python3 build_col_minors_cache_playwright.py --headed --interactive
  python3 build_col_minors_cache_playwright.py --season 2026 --out col_minors_cache.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from playwright.async_api import async_playwright


BASE = "https://www.fangraphs.com/api/leaders/minor-league/data"
LG = "2,4,5,6,7,8,9,10,11,14,12,13,15,16,17,18,30,32"
ORG = "19"  # Rockies


def build_url(
    season: int,
    stats: str,
    type_: int,
    *,
    qual: str = "0",
    include_lg: bool = True,
    include_org: bool = True,
) -> str:
    lg_part = f"&lg={LG}" if include_lg else ""
    org_part = f"&org={ORG}" if include_org else ""
    return (
        f"{BASE}?pos=all&level=0{lg_part}&qual={qual}&season={season}&seasonEnd={season}"
        f"{org_part}&ind=0&splitTeam=false&players=&stats={stats}&type={type_}"
    )


REQUESTS = {
    "pa1": ("pit", 1),
    "pa2": ("pit", 2),
    "pa4": ("pit", 4),
    "ba1": ("bat", 1),
    "ba2": ("bat", 2),
    "ba4": ("bat", 4),
}

def count_rows(payload: dict | list) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        d = payload.get("data")
        if isinstance(d, list):
            return len(d)
    return 0


async def fetch_json_via_page(page, url: str) -> dict:
    script = """
    async (u) => {
      try {
        const r = await fetch(u, {
          method: 'GET',
          credentials: 'include',
          headers: { 'Accept': 'application/json, text/plain, */*' }
        });
        const text = await r.text();
        return { ok: r.ok, status: r.status, text };
      } catch (e) {
        return { ok: false, status: 0, text: String(e) };
      }
    }
    """
    result = await page.evaluate(script, url)
    if not result.get("ok"):
        raise RuntimeError(f"HTTP {result.get('status')} for {url} :: {result.get('text', '')[:200]}")
    txt = result.get("text", "")
    try:
        return json.loads(txt)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Non-JSON response for {url}: {txt[:200]}") from exc


async def run(season: int, out_path: Path, headed: bool, interactive: bool) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=not headed)
        context = await browser.new_context()
        page = await context.new_page()

        # Warm-up page to let any anti-bot challenge run in browser context.
        await page.goto("https://www.fangraphs.com/", wait_until="domcontentloaded")
        await page.wait_for_timeout(3000)
        if interactive:
            await page.goto(
                "https://www.fangraphs.com/leaders/minor-league"
                f"?pos=all&stats=pit&qual=0&type=1&season={season}&team={ORG}",
                wait_until="domcontentloaded",
            )
            print(
                "\nIf a Cloudflare check appears, complete it in the opened browser window.\n"
                "Then press ENTER in this terminal to continue..."
            )
            input()

        payloads: dict[str, dict] = {}
        for key, (stats, type_) in REQUESTS.items():
            # FanGraphs sometimes returns 500 for certain param combos.
            # Try a few safe variants before failing.
            candidate_urls = [
                build_url(season, stats, type_, qual="0", include_lg=True, include_org=True),
                build_url(season, stats, type_, qual="0", include_lg=False, include_org=True),
                build_url(season, stats, type_, qual="1", include_lg=True, include_org=True),
                build_url(season, stats, type_, qual="0", include_lg=True, include_org=False),
            ]
            data = None
            last_err = None
            for url in candidate_urls:
                for _ in range(2):
                    try:
                        data = await fetch_json_via_page(page, url)
                        break
                    except Exception as exc:  # noqa: BLE001
                        last_err = exc
                        await page.wait_for_timeout(1200)
                if data is not None:
                    break
            if data is None:
                raise RuntimeError(f"Failed {key} after retries: {last_err}")
            payloads[key] = data
            nrows = count_rows(data)
            print(f"{key}: {nrows} rows")

        out_path.write_text(json.dumps({"payloads": payloads}, separators=(",", ":")))
        print(f"Wrote: {out_path.resolve()}")
        await browser.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--out", default="col_minors_cache.json")
    parser.add_argument("--headed", action="store_true", help="Run browser with UI (helps when challenge appears)")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Pause for manual Cloudflare clearance before API pulls",
    )
    args = parser.parse_args()

    out = Path(args.out).expanduser().resolve()
    asyncio.run(run(args.season, out, args.headed, args.interactive))


if __name__ == "__main__":
    main()
