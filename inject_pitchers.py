"""
inject_pitchers.py
──────────────────
After running fetch_pitchers.py, run this to splice the new
pitcher data into coors_movement_sim.html.

Usage:
    python inject_pitchers.py
"""

import re
import sys
from pathlib import Path

JS_FILE   = Path('pitchers_data.js')
HTML_FILE = Path('coors_movement_sim.html')
OUT_FILE  = Path('coors_movement_sim.html')

def main():
    if not JS_FILE.exists():
        print(f"ERROR: {JS_FILE} not found. Run fetch_pitchers.py first.")
        sys.exit(1)
    if not HTML_FILE.exists():
        print(f"ERROR: {HTML_FILE} not found. Make sure it's in the same folder.")
        sys.exit(1)

    new_js   = JS_FILE.read_text()
    html     = HTML_FILE.read_text()

    # Replace everything between the two sentinel comments
    pattern = r'(\/\* ── PITCHER DATA ─+[^\*]*\*\/\n)' \
              r'const PITCHERS = \{.*?\};' \
              r'(\n\nconst TEAMS)'
    
    # Fallback: find const PITCHERS block directly
    start_marker = 'const PITCHERS = {'
    end_marker   = '};\n\nconst TEAMS'

    si = html.find(start_marker)
    ei = html.find(end_marker)

    if si == -1 or ei == -1:
        print("ERROR: Could not find PITCHERS block in HTML. Check the file.")
        sys.exit(1)

    before = html[:si]
    after  = html[ei + len('};'):]   # keep from '};\n\nconst TEAMS' onward

    new_html = before + new_js + after

    OUT_FILE.write_text(new_html)

    # Count pitchers
    count = new_js.count("team:'")
    print(f"✓ Injected {count} pitchers into {OUT_FILE}")
    print(f"  File size: {OUT_FILE.stat().st_size:,} bytes")
    print(f"\nNext:")
    print(f"  cp {OUT_FILE} ~/path/to/rockies-milb-dashboard/")
    print(f"  cd ~/path/to/rockies-milb-dashboard")
    print(f"  git add coors_movement_sim.html")
    print(f'  git commit -m "Full 2025 Statcast roster data - all active pitchers"')
    print(f"  git push origin main")

if __name__ == '__main__':
    main()
