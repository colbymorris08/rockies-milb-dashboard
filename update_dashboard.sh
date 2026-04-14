#!/bin/zsh
set -e
cd ~/rockies

# rebuild cache (uses your current method)
PLAYWRIGHT_BROWSERS_PATH=0 python3 build_col_minors_cache_playwright.py --season 2026 --headed --interactive --out col_minors_cache.json

# commit + push if changed
git add col_minors_cache.json
git diff --cached --quiet || git commit -m "Auto-update dashboard data"
git push
