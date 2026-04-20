"""
fetch_pitchers.py
─────────────────
Pulls current MLB active rosters + 2025 Statcast pitch data for every pitcher.
Outputs: pitchers_data.js  (drop into coors_movement_sim.html)

Requirements:
    pip install pybaseball requests pandas

Usage:
    python fetch_pitchers.py

Runtime: ~10-20 min depending on roster size (Statcast API rate limits).
Uses cached data where possible.
"""

import requests
import pandas as pd
import json
import time
import sys
from pybaseball import statcast_pitcher, playerid_lookup
from pybaseball import cache

cache.enable()

# ── MLB TEAM IDS (Stats API) ──────────────────────────────────
TEAMS = {
    133:'OAK',134:'PIT',135:'SDP',136:'SEA',137:'SFG',138:'STL',
    139:'TBR',140:'TEX',141:'TOR',142:'MIN',143:'PHI',144:'ATL',
    145:'CWS',146:'MIA',147:'NYY',158:'MIL',108:'LAA',109:'ARI',
    110:'BAL',111:'BOS',112:'CHC',113:'CIN',114:'CLE',115:'COL',
    116:'DET',117:'HOU',118:'KCR',119:'LAD',120:'WSN',121:'NYM',
}

PITCH_NAME = {
    'FF':'4-Seam','FA':'Fastball','SI':'Sinker','FT':'2-Seam',
    'FC':'Cutter','CH':'Changeup','SC':'Changeup','FS':'Splitter',
    'FO':'Forkball','CU':'Curveball','KC':'Knuckle Curve','CS':'Slow Curve',
    'SL':'Slider','ST':'Sweeper','SV':'Slurve','KN':'Knuckleball',
    'EP':'Eephus',
}

MIN_PITCHES = 50   # minimum pitches thrown to include pitch type
MIN_IP      = 5    # minimum innings to include pitcher

def get_roster(team_id: int) -> list[dict]:
    """Fetch active 40-man roster from MLB Stats API."""
    url = (f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster"
           f"?season=2026&rosterType=active")
    try:
        r = requests.get(url, timeout=15)
        data = r.json()
        return data.get('roster', [])
    except Exception as e:
        print(f"  Roster fetch failed for team {team_id}: {e}")
        return []

def get_pitcher_mlbam(player_name: str, mlbam_id: int) -> int:
    """Return the MLBAM id (already known from roster)."""
    return mlbam_id

def fetch_statcast_for_pitcher(mlbam_id: int, name: str) -> pd.DataFrame | None:
    """Pull 2025 season Statcast data for a pitcher."""
    try:
        df = statcast_pitcher('2025-03-20', '2025-10-05', player_id=mlbam_id)
        if df is None or len(df) == 0:
            # Try 2024 as fallback
            df = statcast_pitcher('2024-03-20', '2024-10-01', player_id=mlbam_id)
        if df is None or len(df) == 0:
            return None
        return df
    except Exception as e:
        print(f"    Statcast error for {name}: {e}")
        return None

def process_pitcher_data(df: pd.DataFrame, name: str, team: str, throws: str) -> dict | None:
    """
    Aggregate Statcast rows into per-pitch-type averages.
    Returns dict ready for JS PITCHERS object or None if insufficient data.
    """
    needed = ['pitch_type','release_speed','pfx_x','pfx_z',
              'release_spin_rate','release_extension','release_pos_z']
    df = df.dropna(subset=needed)
    df = df[df['pitch_type'].isin(PITCH_NAME.keys())]

    if len(df) == 0:
        return None

    # Check innings (approximate: TBF / 4.3)
    tbf = len(df[df['events'].notna()])
    if tbf < MIN_IP * 4:
        return None

    pitches = []
    total_pitches = len(df)

    for ptype, grp in df.groupby('pitch_type'):
        if len(grp) < MIN_PITCHES:
            continue

        velo  = round(grp['release_speed'].mean(), 1)
        pfx_x = round(grp['pfx_x'].mean() * 12, 2)   # feet → inches
        pfx_z = round(grp['pfx_z'].mean() * 12, 2)   # feet → inches
        spin  = int(grp['release_spin_rate'].mean())
        ext   = round(grp['release_extension'].mean(), 1)
        rh    = round(grp['release_pos_z'].mean(), 1)
        usage = round(len(grp) / total_pitches, 3)

        # pfx_x from Statcast is in catcher's view (positive = glove side for RHP)
        # We store as catcher's view (the HTML handles pitcher's view flip)
        pitches.append([ptype, velo, pfx_x, pfx_z, spin, ext, rh, usage])

    if not pitches:
        return None

    # Sort by usage descending
    pitches.sort(key=lambda x: x[7], reverse=True)

    return {
        'team': team,
        'throws': throws,
        'pitches': pitches,
    }

def build_js(all_pitchers: dict) -> str:
    """Convert pitcher dict to JavaScript const."""
    lines = ['const PITCHERS = {']
    for name, data in sorted(all_pitchers.items()):
        pitch_parts = []
        for p in data['pitches']:
            pitch_parts.append(
                f"['{p[0]}',{p[1]},{p[2]},{p[3]},{p[4]},{p[5]},{p[6]},{p[7]}]"
            )
        pitches_str = ', '.join(pitch_parts)
        safe_name = name.replace("'", "\\'")
        lines.append(
            f"  '{safe_name}':{{team:'{data['team']}',throws:'{data['throws']}',"
            f"pitches:[{pitches_str}]}},"
        )
    lines.append('};')
    return '\n'.join(lines)

def main():
    all_pitchers = {}
    team_counts  = {}

    print(f"\n{'═'*60}")
    print("  COORS MOVEMENT SIM — ROSTER + STATCAST FETCHER")
    print(f"{'═'*60}\n")

    for team_id, abbr in TEAMS.items():
        print(f"\n[{abbr}] Fetching roster...")
        roster = get_roster(team_id)

        pitchers_on_team = [
            p for p in roster
            if p.get('position', {}).get('code') == '1'   # position code 1 = pitcher
        ]
        print(f"  → {len(pitchers_on_team)} pitchers on active roster")

        team_count = 0
        for player in pitchers_on_team:
            pid    = player['person']['id']
            pname  = player['person']['fullName']
            throws = player.get('person', {}).get('pitchHand', {}).get('code', 'R')

            print(f"  Fetching: {pname} ({pid})...", end=' ', flush=True)

            df = fetch_statcast_for_pitcher(pid, pname)
            if df is None:
                print("no data")
                continue

            result = process_pitcher_data(df, pname, abbr, throws)
            if result is None:
                print("insufficient")
                continue

            n_types = len(result['pitches'])
            print(f"✓ {n_types} pitch types, {len(df)} pitches")
            all_pitchers[pname] = result
            team_count += 1
            time.sleep(0.3)  # be polite to the API

        team_counts[abbr] = team_count
        print(f"  [{abbr}] Done: {team_count} pitchers added")

    print(f"\n{'═'*60}")
    print(f"  COMPLETE: {len(all_pitchers)} pitchers across 30 teams")
    for abbr, count in sorted(team_counts.items()):
        print(f"    {abbr}: {count}")
    print(f"{'═'*60}\n")

    # Write JS
    js_content = build_js(all_pitchers)
    with open('pitchers_data.js', 'w') as f:
        f.write(js_content)
    print("✓ Written: pitchers_data.js")

    # Write JSON backup
    with open('pitchers_data.json', 'w') as f:
        json.dump(all_pitchers, f, indent=2)
    print("✓ Written: pitchers_data.json (backup)\n")

    print("Next step:")
    print("  python inject_pitchers.py")
    print("  (or manually paste pitchers_data.js into coors_movement_sim.html)\n")

if __name__ == '__main__':
    main()
