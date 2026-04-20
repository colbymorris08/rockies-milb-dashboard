#!/usr/bin/env python3
"""
Build col_minors_cache.json from manually exported FanGraphs CSV files.

Expected CSV files (from FanGraphs minor-league leaders exports):
  - pa1.csv  -> Pitching, type=1 (advanced)
  - pa2.csv  -> Pitching, type=2 (batted ball)
  - pa4.csv  -> Pitching, type=4 (plate discipline / info)
  - ba1.csv  -> Batting,  type=1 (advanced)
  - ba2.csv  -> Batting,  type=2 (batted ball)
  - ba4.csv  -> Batting,  type=4 (plate discipline / info)

Usage:
  python3 build_col_minors_cache_from_csv.py --dir /path/to/csvs
  python3 build_col_minors_cache_from_csv.py --dir . --out col_minors_cache.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED = ["pa1.csv", "pa2.csv", "pa4.csv", "ba1.csv", "ba2.csv", "ba4.csv"]


def read_csv(path: Path) -> dict:
    df = pd.read_csv(path)
    # Normalize NaN/inf for JSON serialization
    df = df.replace({pd.NA: None})
    rows = json.loads(df.to_json(orient="records"))
    return {"data": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=".", help="Directory containing the 6 CSV files")
    parser.add_argument("--out", default="col_minors_cache.json", help="Output JSON file path")
    args = parser.parse_args()

    src_dir = Path(args.dir).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    missing = [f for f in REQUIRED if not (src_dir / f).is_file()]
    if missing:
        raise SystemExit(
            "Missing required CSV file(s): "
            + ", ".join(missing)
            + "\nPlace all 6 exports in one folder and retry."
        )

    payloads = {
        "pa1": read_csv(src_dir / "pa1.csv"),
        "pa2": read_csv(src_dir / "pa2.csv"),
        "pa4": read_csv(src_dir / "pa4.csv"),
        "ba1": read_csv(src_dir / "ba1.csv"),
        "ba2": read_csv(src_dir / "ba2.csv"),
        "ba4": read_csv(src_dir / "ba4.csv"),
    }

    out_path.write_text(json.dumps({"payloads": payloads}, separators=(",", ":")))
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
