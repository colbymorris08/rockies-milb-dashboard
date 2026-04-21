"""
generate_coors_stuff.py
Generates sea-level Stuff+ and Coors-adjusted Stuff+ scores.
Ranks pitchers using a performance-heavy composite built from:
  - Stuff model scores on the existing model window (2021-2024)
  - Coors performance over the last 10 Statcast seasons (2015-2024)
Outputs JSON files for the dashboard website
"""

import numpy as np
import pandas as pd
import pybaseball as pb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import gc
import json, os, warnings
import time
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Any
warnings.filterwarnings("ignore")

pb.cache.enable()

MODEL_START = "2021-01-01"
MODEL_END = "2024-12-31"
COORS_PERF_START = "2015-01-01"
COORS_PERF_END = "2024-12-31"
MIN_COORS_IP_LAST10 = 30.0
PERF_WEIGHT = 0.50
STUFF_WEIGHT = 0.50

# ── Coors alpha multipliers ────────────────────────────────────────────────
ALPHA = {
    "release_speed":     1.05,
    "pfx_z":             0.79,
    "pfx_x_magnus":      0.82,
    "pfx_x_gyro":        0.97,
    "release_spin_rate": 0.86,
    "log_ext":           1.18,
    "vaa":               1.14,
    "haa":               1.06,
    "delta_velo":        1.05,
    "delta_ivb":         0.79,
    "delta_hb":          0.90,
    "release_pos_z":     1.08,
    "release_pos_x":     1.00,
}
PARK_INTERCEPT = 0.38 / 9
FIP_LEAGUE_C = 3.10  # constant term for FIP on same scale as typical MLB FIP

# Statcast `events` → outs recorded on that play (used when outs_when_up delta is 0)
EVENT_OUTS: dict[str, int] = {
    "strikeout": 1,
    "field_out": 1,
    "force_out": 1,
    "sac_fly": 1,
    "sac_bunt": 1,
    "grounded_into_double_play": 2,
    "double_play": 2,
    "triple_play": 3,
    "fielders_choice_out": 1,
    "field_error": 0,
    "home_run": 0,
    "walk": 0,
    "intent_walk": 0,
    "hit_by_pitch": 0,
    "single": 0,
    "double": 0,
    "triple": 0,
}


def fangraphs_name_to_statcast(name: str) -> str:
    """FanGraphs 'Andrew Abbott' → Statcast 'Abbott, Andrew'."""
    s = str(name).strip()
    if not s:
        return s
    parts = s.split()
    if len(parts) == 1:
        return parts[0]
    return f"{parts[-1]}, {' '.join(parts[:-1])}"


def _coors_pa_totals_from_pitch_data(coors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per pitcher, aggregate plate appearances at Coors (home_team == COL only)
    to get IP (from outs), runs allowed in those PAs, K/BB/HR, FIP, and K−BB%.
    Note: 'coors_ra9' is runs per 9 (Statcast plate runs), not official earned runs.
    """
    d = coors_df.copy()
    if d.empty or "player_name" not in d.columns:
        return pd.DataFrame(
            columns=["player_name", "coors_bf", "coors_ip", "coors_ra9", "coors_fip", "coors_k_bb_pct"]
        )
    need = {
        "game_pk",
        "inning",
        "inning_topbot",
        "at_bat_number",
        "pitch_number",
        "events",
        "player_name",
        "away_score",
        "home_score",
        "post_away_score",
        "post_home_score",
        "outs_when_up",
    }
    miss = need - set(d.columns)
    if miss:
        print(f"  [coors rates] missing columns {miss}, skipping Coors PA aggregates")
        return pd.DataFrame(
            columns=["player_name", "coors_bf", "coors_ip", "coors_ra9", "coors_fip", "coors_k_bb_pct"]
        )

    d = d.sort_values(["game_pk", "inning", "inning_topbot", "at_bat_number", "pitch_number"])
    keys = ["game_pk", "inning", "inning_topbot", "at_bat_number"]
    rows: list[dict[str, Any]] = []
    for _, grp in d.groupby(keys, sort=False):
        g = grp.sort_values("pitch_number")
        last = g.iloc[-1]
        ev = last.get("events")
        if pd.isna(ev) or str(ev).strip() in ("", "truncated_pa"):
            continue
        evs = str(ev)
        first = g.iloc[0]
        pname = str(last["player_name"]).strip()
        if last["inning_topbot"] == "Top":
            runs = float(last["post_away_score"]) - float(first["away_score"])
        else:
            runs = float(last["post_home_score"]) - float(first["home_score"])
        delta_o = max(0, int(last["outs_when_up"]) - int(first["outs_when_up"]))
        eo = int(EVENT_OUTS.get(evs, 0))
        outs = max(delta_o, eo)
        k = 1 if evs == "strikeout" else 0
        bb = 1 if evs in ("walk", "intent_walk") else 0
        hbp = 1 if evs == "hit_by_pitch" else 0
        hr = 1 if evs == "home_run" else 0
        rows.append(
            {
                "player_name": pname,
                "runs": runs,
                "outs": outs,
                "k": k,
                "bb": bb,
                "hbp": hbp,
                "hr": hr,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["player_name", "coors_bf", "coors_ip", "coors_ra9", "coors_fip", "coors_k_bb_pct"]
        )
    pa = pd.DataFrame(rows)
    agg = (
        pa.groupby("player_name", as_index=False)
        .agg(
            runs=("runs", "sum"),
            outs=("outs", "sum"),
            k=("k", "sum"),
            bb=("bb", "sum"),
            hbp=("hbp", "sum"),
            hr=("hr", "sum"),
            coors_bf=("runs", "count"),
        )
    )
    agg["coors_ip"] = agg["outs"] / 3.0
    agg["coors_ra9"] = np.where(agg["coors_ip"] > 0, agg["runs"] * 9.0 / agg["coors_ip"], np.nan)
    agg["coors_fip"] = np.where(
        agg["coors_ip"] > 0,
        (13.0 * agg["hr"] + 3.0 * (agg["bb"] + agg["hbp"]) - 2.0 * agg["k"]) / agg["coors_ip"] + FIP_LEAGUE_C,
        np.nan,
    )
    agg["coors_k_bb_pct"] = np.where(
        agg["coors_bf"] > 0, (agg["k"] - agg["bb"]) / agg["coors_bf"] * 100.0, np.nan
    )
    return agg[["player_name", "runs", "outs", "k", "bb", "hbp", "hr", "coors_bf"]].copy()


def _rates_from_totals(totals_df: pd.DataFrame) -> pd.DataFrame:
    if totals_df.empty:
        return pd.DataFrame(
            columns=["player_name", "coors_bf", "coors_ip", "coors_ra9", "coors_fip", "coors_k_bb_pct"]
        )
    agg = totals_df.copy()
    agg["coors_ip"] = agg["outs"] / 3.0
    agg["coors_ra9"] = np.where(agg["coors_ip"] > 0, agg["runs"] * 9.0 / agg["coors_ip"], np.nan)
    agg["coors_fip"] = np.where(
        agg["coors_ip"] > 0,
        (13.0 * agg["hr"] + 3.0 * (agg["bb"] + agg["hbp"]) - 2.0 * agg["k"]) / agg["coors_ip"] + FIP_LEAGUE_C,
        np.nan,
    )
    agg["coors_k_bb_pct"] = np.where(
        agg["coors_bf"] > 0, (agg["k"] - agg["bb"]) / agg["coors_bf"] * 100.0, np.nan
    )
    out = agg[["player_name", "coors_bf", "coors_ip", "coors_ra9", "coors_fip", "coors_k_bb_pct"]].copy()
    return out.round({"coors_ip": 1, "coors_ra9": 2, "coors_fip": 2, "coors_k_bb_pct": 1})


def compute_coors_pitching_rates(coors_df: pd.DataFrame) -> pd.DataFrame:
    totals = _coors_pa_totals_from_pitch_data(coors_df)
    return _rates_from_totals(totals)


def safe_statcast_pull(start_dt: str, end_dt: str, team: str | None = None, tries: int = 4) -> pd.DataFrame:
    """
    Retry wrapper around pybaseball.statcast to handle intermittent malformed CSV
    responses (e.g., pandas ParserError from partial/non-CSV payloads).
    """
    last_err: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            return pb.statcast(start_dt=start_dt, end_dt=end_dt, team=team, parallel=False)
        except Exception as e:
            last_err = e
            print(f"    statcast pull failed {start_dt}..{end_dt} (attempt {attempt}/{tries}): {e}")
            if attempt < tries:
                time.sleep(2.0 * attempt)
    raise RuntimeError(f"statcast pull failed after {tries} attempts for {start_dt}..{end_dt}") from last_err


def _daterange_chunks(start_dt: str, end_dt: str, chunk_days: int) -> list[tuple[str, str]]:
    s = datetime.strptime(start_dt, "%Y-%m-%d").date()
    e = datetime.strptime(end_dt, "%Y-%m-%d").date()
    out: list[tuple[str, str]] = []
    cur = s
    while cur < e:
        nxt = min(cur + timedelta(days=chunk_days), e)
        out.append((cur.isoformat(), nxt.isoformat()))
        cur = nxt
    return out


def safe_statcast_pull_adaptive(start_dt: str, end_dt: str, team: str | None = None) -> pd.DataFrame:
    """
    Pull Statcast robustly:
      1) try full window
      2) on timeout/failure, split into 14-day chunks
      3) if a 14-day chunk still fails, split that chunk into 7-day chunks
      4) if a 7-day chunk still fails, split that chunk into 1-day chunks
    """
    try:
        return safe_statcast_pull(start_dt, end_dt, team=team)
    except Exception as first_err:
        print(f"    adaptive split for {start_dt}..{end_dt} after error: {first_err}")

    pieces_14: list[pd.DataFrame] = []
    for s14, e14 in _daterange_chunks(start_dt, end_dt, 14):
        try:
            d14 = safe_statcast_pull(s14, e14, team=team)
            if not d14.empty:
                pieces_14.append(d14)
            continue
        except Exception as err14:
            print(f"    14-day chunk failed {s14}..{e14}: {err14}; retrying weekly")

        for s7, e7 in _daterange_chunks(s14, e14, 7):
            try:
                d7 = safe_statcast_pull(s7, e7, team=team)
                if not d7.empty:
                    pieces_14.append(d7)
                continue
            except Exception as err7:
                print(f"    7-day chunk failed {s7}..{e7}: {err7}; retrying daily")

            for s1, e1 in _daterange_chunks(s7, e7, 1):
                try:
                    d1 = safe_statcast_pull(s1, e1, team=team)
                    if not d1.empty:
                        pieces_14.append(d1)
                except Exception as err1:
                    # Do not abort full model generation on a stubborn day-sized timeout.
                    print(f"    daily chunk failed {s1}..{e1}: {err1}; skipping day")

    if not pieces_14:
        return pd.DataFrame()
    return pd.concat(pieces_14, ignore_index=True)


def pull_statcast_yearly(start_year: int, end_year: int, team: str | None = None) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        print(f"  pulling Statcast season {year}...")
        month_chunks: list[pd.DataFrame] = []
        for month in range(1, 13):
            start_dt = f"{year}-{month:02d}-01"
            if month == 12:
                end_dt = f"{year}-12-31"
            else:
                end_dt = f"{year}-{month+1:02d}-01"
            y = safe_statcast_pull_adaptive(start_dt, end_dt, team=team)
            if not y.empty:
                month_chunks.append(y)
        if month_chunks:
            chunks.append(pd.concat(month_chunks, ignore_index=True))
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def pull_coors_last10_rates_efficient(start_year: int, end_year: int) -> pd.DataFrame:
    """
    Memory-safe Coors pull:
      1) pull one season at a time
      2) immediately filter to COL home games
      3) aggregate to PA totals and discard raw pitch table
      4) prefilter to pitchers with at least 2 Coors BF
      5) final filter to MIN_COORS_IP_LAST10 for ranking eligibility
    """
    season_totals: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        print(f"    pulling Coors season {year}...")
        # Full-season single query can timeout; use adaptive chunked pulls.
        y = safe_statcast_pull_adaptive(f"{year}-01-01", f"{year}-12-31")
        if y.empty:
            print(f"    no Statcast rows for {year}; skipping")
            continue
        y = y[(y["game_type"] == "R") & (y["home_team"] == "COL")].copy()
        t = _coors_pa_totals_from_pitch_data(y)
        if not t.empty:
            season_totals.append(t)
        del y
        del t
        gc.collect()

    if not season_totals:
        return pd.DataFrame(
            columns=["player_name", "coors_bf", "coors_ip", "coors_ra9", "coors_fip", "coors_k_bb_pct"]
        )

    totals = (
        pd.concat(season_totals, ignore_index=True)
        .groupby("player_name", as_index=False)[["runs", "outs", "k", "bb", "hbp", "hr", "coors_bf"]]
        .sum()
    )
    # Early sample-size filter per your request.
    totals = totals[totals["coors_bf"] >= 2].copy()
    rates = _rates_from_totals(totals)
    rates = rates[rates["coors_ip"] >= MIN_COORS_IP_LAST10].copy()
    return rates


# ── 1. Pull Statcast model window (2021-2024) ──────────────────────────────
print("Pulling Statcast data...")
df = pull_statcast_yearly(2021, 2024)
df = df[df["game_type"] == "R"].copy()
print(f"  {len(df):,} pitches")

# ── 2. Feature engineering ─────────────────────────────────────────────────
def engineer(d):
    d = d.copy()
    fb = d[d["pitch_type"].isin(["FF","SI"])].groupby("pitcher")[
        ["release_speed","pfx_z","pfx_x"]
    ].mean().rename(columns={"release_speed":"fb_velo","pfx_z":"fb_ivb","pfx_x":"fb_hb"})
    d = d.merge(fb, on="pitcher", how="left")
    d["delta_velo"]    = d["release_speed"] - d["fb_velo"]
    d["delta_ivb"]     = d["pfx_z"]         - d["fb_ivb"]
    d["delta_hb"]      = d["pfx_x"]         - d["fb_hb"]
    d["log_ext"]       = np.log(d["release_extension"].clip(lower=0.1))
    d["vaa"]           = np.degrees(np.arctan2(d["pfx_z"], 60.5))
    d["haa"]           = np.degrees(np.arctan2(d["pfx_x"], 60.5))
    gyro = ["SL","ST","FC","CU"]
    d["pfx_x_gyro"]   = np.where(d["pitch_type"].isin(gyro), d["pfx_x"], 0.0)
    d["pfx_x_magnus"] = np.where(~d["pitch_type"].isin(gyro), d["pfx_x"], 0.0)
    d["stand_enc"]    = (d["stand"] == "R").astype(int)
    return d

df = engineer(df)
le_pitch = LabelEncoder()
df["pitch_type_enc"] = le_pitch.fit_transform(df["pitch_type"].fillna("XX"))

FEATURES = [
    "release_speed","pfx_z","pfx_x_magnus","pfx_x_gyro",
    "release_spin_rate","log_ext","delta_velo","delta_ivb","delta_hb",
    "release_pos_z","release_pos_x",
    "balls","strikes","stand_enc","pitch_type_enc",
    "vaa","haa","spin_axis",
]
FEATURES = [f for f in FEATURES if f in df.columns]
TARGET = "estimated_woba_using_speedangle"

# ── 4. Coors vs. other split ───────────────────────────────────────────────
coors = df[df["home_team"] == "COL"].copy()
other = df[df["home_team"] != "COL"].copy()

def apply_alpha(d, alpha_map):
    d = d.copy()
    for feat, alpha in alpha_map.items():
        if feat in d.columns:
            d[feat] = d[feat] * alpha
    return d

coors_adj = apply_alpha(coors, ALPHA)

# ── 5. Train TWO models: sea-level and Coors-adjusted ─────────────────────
combined_sl  = pd.concat([coors, other], ignore_index=True).dropna(subset=FEATURES+[TARGET])
combined_col = pd.concat([coors_adj, other], ignore_index=True).dropna(subset=FEATURES+[TARGET])

def fit_model(X, y, label):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=99)
    m = XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=99, n_jobs=-1, verbosity=0,
        eval_metric="rmse", early_stopping_rounds=20,
    )
    m.fit(Xtr, ytr, eval_set=[(Xte, yte)], verbose=False)
    r2  = m.score(Xte, yte)
    rmse = root_mean_squared_error(yte, m.predict(Xte))
    print(f"  {label}: Test R²={r2:.4f}  RMSE={rmse:.4f}  trees={m.best_iteration}")
    return m

print("\nFitting sea-level model...")
m_sl  = fit_model(combined_sl[FEATURES],  combined_sl[TARGET],  "Sea-level")
print("Fitting Coors-adjusted model...")
m_col = fit_model(combined_col[FEATURES], combined_col[TARGET], "Coors-adj")

# ── 6. Score every Coors pitch under both models ───────────────────────────
score_base = coors.dropna(subset=FEATURES).copy()
score_adj  = coors_adj.loc[score_base.index].copy()

score_base["xRV_sl"]  = m_sl.predict(score_base[FEATURES])
score_adj["xRV_col"]  = m_col.predict(score_adj[FEATURES]) + PARK_INTERCEPT

# standardize both to 100/10 scale
def standardize(s):
    return 100 + (s - s.mean()) / s.std() * 10

score_base["stuff_sl"]  = standardize(score_base["xRV_sl"])
score_adj["stuff_col"]  = standardize(score_adj["xRV_col"])

combined_scores = score_base[["player_name","pitch_type","game_date","release_speed",
                               "pfx_z","pfx_x","release_spin_rate","release_extension",
                               "vaa","haa","stuff_sl","xRV_sl"]].copy()
combined_scores["stuff_col"] = score_adj["stuff_col"].values
combined_scores["xRV_col"]   = score_adj["xRV_col"].values
combined_scores["stuff_delta"] = combined_scores["stuff_col"] - combined_scores["stuff_sl"]

# At Coors (home_team == COL), Top half = away batting = Rockies pitching.
if "inning_topbot" in coors.columns:
    _ibt = coors.loc[score_base.index, "inning_topbot"].astype(str).str.upper()
    combined_scores["is_rockies_pitch"] = (_ibt == "TOP").values
else:
    combined_scores["is_rockies_pitch"] = pd.Series(False, index=combined_scores.index)

MIN_PITCHES_ALL = 1
MIN_PITCHES_ROCKIES = 1

# ── 7. Pitcher-level aggregation ───────────────────────────────────────────
def aggregate_pitchers(sub: pd.DataFrame, min_pitches: int) -> pd.DataFrame:
    return (
        sub.groupby("player_name")
        .agg(
            pitches=("stuff_sl", "count"),
            stuff_sl=("stuff_sl", "mean"),
            stuff_col=("stuff_col", "mean"),
            stuff_delta=("stuff_delta", "mean"),
            avg_velo=("release_speed", "mean"),
            avg_ivb=("pfx_z", "mean"),
            avg_hb=("pfx_x", "mean"),
            avg_spin=("release_spin_rate", "mean"),
            avg_ext=("release_extension", "mean"),
            avg_vaa=("vaa", "mean"),
        )
        .reset_index()
        .query(f"pitches >= {min_pitches}")
        .round(2)
    )


pitcher_agg = aggregate_pitchers(combined_scores, MIN_PITCHES_ALL)
pitcher_agg_rockies = aggregate_pitchers(
    combined_scores[combined_scores["is_rockies_pitch"]], MIN_PITCHES_ROCKIES
)

print("  Coors Field PA aggregates (Statcast 2021-2024 games at Coors)...")
coors_rates = compute_coors_pitching_rates(coors)
pitcher_agg = pitcher_agg.merge(coors_rates, on="player_name", how="left")
pitcher_agg_rockies = pitcher_agg_rockies.merge(coors_rates, on="player_name", how="left")

print("  Last-10-years Coors PA rates (Statcast 2015-2024, COL home regular season)...")
_career_cols = [
    "player_name",
    "career_coors_bf",
    "career_coors_ip",
    "career_coors_ra9",
    "career_coors_fip",
    "career_coors_k_bb_pct",
]
try:
    career_rates = pull_coors_last10_rates_efficient(2015, 2024).rename(
        columns={
            "coors_bf": "career_coors_bf",
            "coors_ip": "career_coors_ip",
            "coors_ra9": "career_coors_ra9",
            "coors_fip": "career_coors_fip",
            "coors_k_bb_pct": "career_coors_k_bb_pct",
        }
    )
    pitcher_agg = pitcher_agg.merge(career_rates, on="player_name", how="left")
    pitcher_agg_rockies = pitcher_agg_rockies.merge(career_rates, on="player_name", how="left")
    print(f"    merged career Coors lines for {len(career_rates)} pitchers")
except Exception as e:
    print(f"    career Coors pull/merge failed ({e}); exporting without career columns")
    for c in _career_cols[1:]:
        if c not in pitcher_agg.columns:
            pitcher_agg[c] = np.nan
        if c not in pitcher_agg_rockies.columns:
            pitcher_agg_rockies[c] = np.nan

def add_performance_composite(d: pd.DataFrame) -> pd.DataFrame:
    out = d.copy()
    out = out[out["career_coors_ip"].fillna(0) >= MIN_COORS_IP_LAST10].copy()
    if out.empty:
        out["coors_composite"] = np.nan
        return out

    # Prefer xFIP if available from any merged source; otherwise use Coors FIP.
    perf_fip = out["xFIP"] if "xFIP" in out.columns else out["career_coors_fip"]
    perf_kbb = out["career_coors_k_bb_pct"]
    fip_std = perf_fip.std() or 1.0
    kbb_std = perf_kbb.std() or 1.0
    fip_z = (perf_fip.mean() - perf_fip) / fip_std
    kbb_z = (perf_kbb - perf_kbb.mean()) / kbb_std
    perf_score = (fip_z + kbb_z) / 2.0

    stuff_std = out["stuff_col"].std() or 1.0
    stuff_z = (out["stuff_col"] - out["stuff_col"].mean()) / stuff_std
    out["coors_perf_score"] = perf_score.round(4)
    out["coors_composite"] = (
        (STUFF_WEIGHT * stuff_z + PERF_WEIGHT * perf_score) * 10.0 + 100.0
    ).round(1)
    return out

pitcher_agg = add_performance_composite(pitcher_agg)
pitcher_agg = pitcher_agg.sort_values("coors_composite", ascending=False)
top20 = pitcher_agg.head(20).copy()

pitcher_agg_rockies = add_performance_composite(pitcher_agg_rockies)
pitcher_agg_rockies = pitcher_agg_rockies.sort_values("coors_composite", ascending=False)
top20_rockies = pitcher_agg_rockies.head(20).copy()

print(f"\nTop 20 Coors pitchers (all, min {MIN_COORS_IP_LAST10:.0f} Coors IP in 2015-2024):")
_cols = ["player_name", "pitches", "stuff_sl", "stuff_col", "stuff_delta", "avg_velo", "avg_ivb", "coors_composite"]
print(top20[[c for c in _cols if c in top20.columns]].to_string(index=False))

print(f"\nTop Rockies pitchers at Coors (min {MIN_COORS_IP_LAST10:.0f} Coors IP in 2015-2024):")
print(top20_rockies[[c for c in _cols if c in top20_rockies.columns]].to_string(index=False))

# ── 8. Pitch-type breakdown per top pitcher ────────────────────────────────
_breakout_names = set(top20["player_name"]) | set(top20_rockies["player_name"])
pitch_breakdown = (
    combined_scores[combined_scores["player_name"].isin(_breakout_names)]
    .groupby(["player_name", "pitch_type"])
    .agg(
        pitches=("stuff_sl", "count"),
        stuff_sl=("stuff_sl", "mean"),
        stuff_col=("stuff_col", "mean"),
        avg_velo=("release_speed", "mean"),
        avg_ivb=("pfx_z", "mean"),
        avg_hb=("pfx_x", "mean"),
    )
    .reset_index()
    .query("pitches >= 30")
    .round(2)
)

_pbt_by_player = defaultdict(list)
for row in pitch_breakdown.itertuples(index=False):
    _pbt_by_player[str(row.player_name)].append(
        {
            "pitch_type": str(row.pitch_type),
            "stuff_sl": float(row.stuff_sl),
            "stuff_col": float(row.stuff_col),
        }
    )

top20["pitches_by_type"] = top20["player_name"].map(lambda n: _pbt_by_player.get(str(n), []))
top20_rockies["pitches_by_type"] = top20_rockies["player_name"].map(lambda n: _pbt_by_player.get(str(n), []))

# ── 9. Feature importance for both models ─────────────────────────────────
def get_importance(model, label):
    imp = dict(zip(FEATURES, model.feature_importances_))
    return [{"feature": str(k), "importance": float(round(float(v), 4)),
             "alpha": float(ALPHA.get(k,1.0)), "model": str(label)}
            for k,v in sorted(imp.items(), key=lambda x:-x[1])]

importance_sl  = get_importance(m_sl,  "sea_level")
importance_col = get_importance(m_col, "coors_adj")

def _safe_mean(series: pd.Series, ndigits: int = 2) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    return float(round(float(s.mean()), ndigits))

# ── 10. Model metadata ─────────────────────────────────────────────────────
meta = {
    "generated": pd.Timestamp.now().isoformat(),
    "training_years": "2021-2024",
    "career_coors_statcast_years": "2015-2024",
    "features": FEATURES,
    "models": ["tjStuff+ v2", "FanGraphs Stuff+/PitchingBot", "aStuff+ v2"],
    "coors_alpha": ALPHA,
    "park_intercept": PARK_INTERCEPT,
    "min_pitches_all_pitchers": MIN_PITCHES_ALL,
    "min_pitches_rockies_at_coors": MIN_PITCHES_ROCKIES,
    "min_coors_ip_last10y_for_ranking": MIN_COORS_IP_LAST10,
    "composite_weights": {"stuff": STUFF_WEIGHT, "coors_performance": PERF_WEIGHT},
    "performance_inputs": "xFIP_if_available_else_FIP_plus_K_minus_BB_percent",
    "top20_names": top20["player_name"].tolist(),
    "rockies_top20_names": top20_rockies["player_name"].tolist(),
    "top20_key_findings": {
        "avg_stuff_delta": _safe_mean(top20["stuff_delta"]),
        "avg_vaa": _safe_mean(top20["avg_vaa"]),
        "avg_ivb": _safe_mean(top20["avg_ivb"]),
        "avg_ext": _safe_mean(top20["avg_ext"]),
    }
}

# ── 11. Export ─────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

top20.to_json("data/coors_top20.json", orient="records", indent=2)
top20_rockies.to_json("data/coors_pitchers_rockies.json", orient="records", indent=2)
pitch_breakdown.to_json("data/coors_pitch_breakdown.json", orient="records", indent=2)

with open("data/coors_importance_sl.json",  "w") as f: json.dump(importance_sl,  f, indent=2)
with open("data/coors_importance_col.json", "w") as f: json.dump(importance_col, f, indent=2)
with open("data/coors_model_meta.json",     "w") as f: json.dump(meta,           f, indent=2)

# pitch-level sample for the simulator (5k pitches, top pitchers only)
sample = combined_scores.dropna().sample(min(5000, len(combined_scores)), random_state=99)
sample.to_json("data/coors_pitch_sample.json", orient="records", indent=2, date_format="iso")

print("\nExported to data/:")
print("  coors_top20.json")
print("  coors_pitchers_rockies.json")
print("  coors_pitch_breakdown.json")
print("  coors_importance_sl.json")
print("  coors_importance_col.json")
print("  coors_model_meta.json")
print("  coors_pitch_sample.json")
print("\nDeploy: git add data/ && git commit -m 'add Coors stuff+ model data' && git push")
