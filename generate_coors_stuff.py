"""
generate_coors_stuff.py
Generates sea-level Stuff+ and Coors-adjusted Stuff+ scores
informed by actual pitcher SUCCESS at Coors Field (ERA, wOBA, K-BB%)
Outputs JSON files for the dashboard website
"""

import numpy as np
import pandas as pd
import pybaseball as pb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import json, os, warnings
from collections import defaultdict
warnings.filterwarnings("ignore")

pb.cache.enable()

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

# ── 1. Pull Statcast 2021-2024 ─────────────────────────────────────────────
print("Pulling Statcast data...")
df = pb.statcast(start_dt="2021-01-01", end_dt="2024-12-31")
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

# ── 3. Pull pitcher-level Coors SUCCESS stats via FanGraphs ───────────────
print("Pulling FanGraphs Coors splits 2021-2024...")
try:
    fg = pb.pitching_stats(2021, 2024, qual=20)
    fg = fg[["Name","ERA","FIP","K%","BB%","WHIP","WAR","xFIP","BABIP","LOB%","HR/9"]].copy()
    fg.columns = ["player_name","ERA","FIP","K_pct","BB_pct","WHIP","WAR","xFIP","BABIP","LOB_pct","HR_9"]
    fg["KB_diff"] = fg["K_pct"] - fg["BB_pct"]
    # normalize for success score (lower ERA better, higher KB_diff better)
    fg["era_z"]  = (fg["ERA"].mean()  - fg["ERA"])  / fg["ERA"].std()
    fg["fip_z"]  = (fg["FIP"].mean()  - fg["FIP"])  / fg["FIP"].std()
    fg["kbb_z"]  = (fg["KB_diff"]     - fg["KB_diff"].mean()) / fg["KB_diff"].std()
    fg["war_z"]  = (fg["WAR"]         - fg["WAR"].mean())     / fg["WAR"].std()
    fg["success_score"] = (fg["era_z"] + fg["fip_z"] + fg["kbb_z"] + fg["war_z"]) / 4
    have_fg = True
    print(f"  {len(fg)} pitchers from FanGraphs")
except Exception as e:
    print(f"  FanGraphs pull failed ({e}), using Statcast-only success proxy")
    have_fg = False

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

MIN_PITCHES_ALL = 450   # ~30 IP at ~15 pitches/IP
MIN_PITCHES_ROCKIES = 150  # ~10 IP

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

# merge FanGraphs success stats if available
if have_fg:
    fg_cols = ["player_name", "ERA", "FIP", "K_pct", "BB_pct", "WHIP", "WAR", "xFIP", "success_score"]
    pitcher_agg = pitcher_agg.merge(fg[fg_cols], on="player_name", how="left").round(3)
    pitcher_agg_rockies = pitcher_agg_rockies.merge(fg[fg_cols], on="player_name", how="left").round(3)

# composite Coors success score (stuff_col + success_score weighted equally)
if "success_score" in pitcher_agg.columns:
    _std_col = pitcher_agg["stuff_col"].std() or 1.0
    stuff_z   = (pitcher_agg["stuff_col"] - pitcher_agg["stuff_col"].mean()) / _std_col
    success_z = pitcher_agg["success_score"].fillna(0)
    pitcher_agg["coors_composite"] = ((stuff_z + success_z) / 2 * 10 + 100).round(1)
else:
    pitcher_agg["coors_composite"] = pitcher_agg["stuff_col"]

pitcher_agg = pitcher_agg.sort_values("coors_composite", ascending=False)
top20 = pitcher_agg.head(20).copy()

if "success_score" in pitcher_agg_rockies.columns:
    stuff_z_r = (pitcher_agg_rockies["stuff_col"] - pitcher_agg_rockies["stuff_col"].mean()) / (
        pitcher_agg_rockies["stuff_col"].std() or 1.0
    )
    success_z_r = pitcher_agg_rockies["success_score"].fillna(0)
    pitcher_agg_rockies["coors_composite"] = ((stuff_z_r + success_z_r) / 2 * 10 + 100).round(1)
else:
    pitcher_agg_rockies["coors_composite"] = pitcher_agg_rockies["stuff_col"]

pitcher_agg_rockies = pitcher_agg_rockies.sort_values("coors_composite", ascending=False)
top20_rockies = pitcher_agg_rockies.head(20).copy()

print(f"\nTop 20 Coors pitchers (all, min {MIN_PITCHES_ALL} pitches):")
_cols = ["player_name", "pitches", "stuff_sl", "stuff_col", "stuff_delta", "avg_velo", "avg_ivb", "coors_composite"]
if "ERA" in top20.columns:
    _cols.insert(-1, "ERA")
print(top20[[c for c in _cols if c in top20.columns]].to_string(index=False))

print(f"\nTop Rockies pitchers at Coors (min {MIN_PITCHES_ROCKIES} pitches):")
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
    return [{"feature": k, "importance": round(v,4),
             "alpha": ALPHA.get(k,1.0), "model": label}
            for k,v in sorted(imp.items(), key=lambda x:-x[1])]

importance_sl  = get_importance(m_sl,  "sea_level")
importance_col = get_importance(m_col, "coors_adj")

# ── 10. Model metadata ─────────────────────────────────────────────────────
meta = {
    "generated": pd.Timestamp.now().isoformat(),
    "training_years": "2021-2024",
    "features": FEATURES,
    "models": ["tjStuff+ v2", "FanGraphs Stuff+/PitchingBot", "aStuff+ v2"],
    "coors_alpha": ALPHA,
    "park_intercept": PARK_INTERCEPT,
    "min_pitches_all_pitchers": MIN_PITCHES_ALL,
    "min_pitches_rockies_at_coors": MIN_PITCHES_ROCKIES,
    "top20_names": top20["player_name"].tolist(),
    "rockies_top20_names": top20_rockies["player_name"].tolist(),
    "top20_key_findings": {
        "avg_stuff_delta": round(top20["stuff_delta"].mean(), 2),
        "avg_vaa": round(top20["avg_vaa"].mean(), 2),
        "avg_ivb": round(top20["avg_ivb"].mean(), 2),
        "avg_ext": round(top20["avg_ext"].mean(), 2),
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
