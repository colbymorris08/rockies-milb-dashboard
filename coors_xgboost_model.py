"""
Coors Field XGBoost Pitch Model
Features drawn from: tjStuff+ v2 (Nestico), FanGraphs Stuff+/PitchingBot (Grove/Sarris), aStuff+ v2 (Salorio)
Altitude adjustments based on Denver air density (82% of sea level)
Target: estimated_woba_using_speedangle (proxy for xRV)
Output: JSON files for GitHub Pages deployment
"""

import numpy as np
import pandas as pd
import pybaseball as pb
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import json
import os

pb.cache.enable()


# ── Coors altitude alpha multipliers ──────────────────────────────────────
# Based on 82% air density at 5,280 ft — Magnus force reduced proportionally
ALPHA = {
    "release_speed":        1.05,  # perceived velo boost — less drag
    "pfx_z":                0.79,  # IVB (Magnus-dominant) — loses 3-4" on 4-seam
    "pfx_x_magnus":         0.82,  # HB Magnus component (sinker/2-seam run)
    "pfx_x_gyro":           0.97,  # HB gyro component (slider/cutter) — gravity-driven, stable
    "release_spin_rate":    0.86,  # raw spin generates less movement in thin air
    "log_ext":              1.18,  # extension amplified — shorter travel distance
    "vaa":                  1.14,  # steep VAA harder to time when movement is suppressed
    "haa":                  1.06,  # horizontal entry angle retains value
    "delta_velo":           1.05,  # speed tunneling more disorienting w/o break
    "delta_ivb":            0.79,  # vertical tunneling reduced — all pitches lose drop
    "delta_hb":             0.90,  # horizontal tunneling partially retained
    "release_pos_z":        1.08,  # vertical release point — overhand slot more effective
    "release_pos_x":        1.00,  # horizontal release — neutral
}

PARK_INTERCEPT  = 0.38 / 9   # ERA baseline lift per IP at Coors
HR_PARK_FACTOR  = 1.42        # HR/FB rate at Coors vs. league average
BABIP_LIFT      = 0.045       # BABIP lift on non-GB contact in thin air


# ── 1. Pull Statcast data (2021-2024 — Hawk-Eye era) ──────────────────────
print("Pulling Statcast data 2021-2024...")
df = pb.statcast(start_dt="2021-01-01", end_dt="2024-12-31")
df = df[df["game_type"] == "R"].copy()
print(f"  {len(df):,} pitches loaded")


# ── 2. Feature engineering ─────────────────────────────────────────────────
def engineer_features(d):
    d = d.copy()

    # primary FB averages per pitcher (4-seam or sinker)
    fb = d[d["pitch_type"].isin(["FF", "SI"])].groupby("pitcher")[
        ["release_speed", "pfx_z", "pfx_x"]
    ].mean().rename(columns={"release_speed": "fb_velo", "pfx_z": "fb_ivb", "pfx_x": "fb_hb"})
    d = d.merge(fb, on="pitcher", how="left")

    # delta features (from primary FB) — core to all three models
    d["delta_velo"] = d["release_speed"] - d["fb_velo"]
    d["delta_ivb"]  = d["pfx_z"]         - d["fb_ivb"]
    d["delta_hb"]   = d["pfx_x"]         - d["fb_hb"]

    # log-transform extension — limits outlier impact (tjStuff+ v2 method)
    d["log_ext"] = np.log(d["release_extension"].clip(lower=0.1))

    # VAA and HAA — approach angles (aStuff+ v2)
    d["vaa"] = np.degrees(np.arctan2(d["pfx_z"], 60.5))
    d["haa"] = np.degrees(np.arctan2(d["pfx_x"], 60.5))

    # gyro vs magnus HB split — gyro (slider/cut) vs magnus (sinker/4S run)
    # approximate: sliders/cutters treated as gyro-dominant
    gyro_types = ["SL", "ST", "FC", "CU"]
    d["pfx_x_gyro"]   = np.where(d["pitch_type"].isin(gyro_types), d["pfx_x"], 0.0)
    d["pfx_x_magnus"] = np.where(~d["pitch_type"].isin(gyro_types), d["pfx_x"], 0.0)

    # batter handedness encode
    d["stand_enc"] = (d["stand"] == "R").astype(int)

    # count state
    d["count_state"] = d["balls"].astype(str) + "-" + d["strikes"].astype(str)

    return d


df = engineer_features(df)
print("  Features engineered")


# ── 3. Encode pitch type and count ─────────────────────────────────────────
le_pitch = LabelEncoder()
le_count = LabelEncoder()
df["pitch_type_enc"] = le_pitch.fit_transform(df["pitch_type"].fillna("XX"))
df["count_state_enc"] = le_count.fit_transform(df["count_state"])


# ── 4. Define full feature set (all three models combined) ─────────────────
# tjStuff+ v2 — velocity, movement, spin, extension, deltas
TJ_FEATURES = [
    "release_speed",        # velocity
    "pfx_z",                # IVB
    "pfx_x_magnus",         # HB Magnus
    "pfx_x_gyro",           # HB gyro
    "release_spin_rate",    # spin rate
    "log_ext",              # log extension
    "delta_velo",           # Δvelo from FB
    "delta_ivb",            # ΔIVB from FB
    "delta_hb",             # ΔHB from FB
]

# FanGraphs Stuff+/PitchingBot — adds release point, location, count, handedness
FG_FEATURES = [
    "release_pos_z",        # vertical release point
    "release_pos_x",        # horizontal release point
    "stand_enc",            # batter handedness
    "pitch_type_enc",       # pitch type categorical
]

# aStuff+ v2 — adds spin axis, VAA, HAA
AS_FEATURES = [
    "vaa",                  # vertical approach angle
    "haa",                  # horizontal approach angle
    "spin_axis",            # spin axis (degrees)
]

ALL_FEATURES = list(dict.fromkeys(TJ_FEATURES + FG_FEATURES + AS_FEATURES))
FEATURES = [f for f in ALL_FEATURES if f in df.columns]
TARGET = "estimated_woba_using_speedangle"


# ── 5. Split Coors vs. non-Coors ───────────────────────────────────────────
coors = df[df["home_team"] == "COL"].copy()
other = df[df["home_team"] != "COL"].copy()
print(f"  Coors pitches: {len(coors):,} | Other: {len(other):,}")


# ── 6. Apply altitude alpha multipliers to Coors feature columns ───────────
def apply_coors_adjustments(d, alpha_map):
    d = d.copy()
    for feat, alpha in alpha_map.items():
        if feat in d.columns:
            d[feat] = d[feat] * alpha
    return d

coors_adj = apply_coors_adjustments(coors, ALPHA)


# ── 7. Combine and clean ───────────────────────────────────────────────────
combined = pd.concat([coors_adj, other], ignore_index=True)
combined = combined.dropna(subset=FEATURES + [TARGET])
print(f"  Training rows after dropna: {len(combined):,}")

X = combined[FEATURES]
y = combined[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=99
)


# ── 8. XGBoost model ───────────────────────────────────────────────────────
print("\nFitting XGBoost...")
model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=5,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,          # L1 regularization
    reg_lambda=1.0,         # L2 regularization
    random_state=99,
    n_jobs=-1,
    verbosity=0,
    eval_metric="rmse",
    early_stopping_rounds=20,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

train_r2  = model.score(X_train, y_train)
test_r2   = model.score(X_test, y_test)
test_rmse = root_mean_squared_error(y_test, model.predict(X_test))

print(f"\nTrain R²: {train_r2:.4f}")
print(f"Test  R²: {test_r2:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

importance = dict(zip(FEATURES, model.feature_importances_))
print("\nFeature importance (descending):")
for k, v in sorted(importance.items(), key=lambda x: -x[1]):
    alpha_note = f"  [α={ALPHA.get(k, 1.0)}]" if k in ALPHA else ""
    print(f"  {k}: {v:.4f}{alpha_note}")


# ── 9. Generate Coors-specific xRV predictions ────────────────────────────
print("\nGenerating Coors predictions...")
coors_pred = coors_adj.copy()
coors_pred = coors_pred.dropna(subset=FEATURES)
coors_pred["xRV_COL"] = model.predict(coors_pred[FEATURES])
coors_pred["xRV_COL"] += PARK_INTERCEPT  # park ERA intercept

# standardize to 100-scale (tjStuff+ convention)
mu  = coors_pred["xRV_COL"].mean()
std = coors_pred["xRV_COL"].std()
coors_pred["xStuff_COL"] = 100 + ((coors_pred["xRV_COL"] - mu) / std) * 10


# ── 10. Pitcher-level summary ──────────────────────────────────────────────
pitcher_summary = (
    coors_pred.groupby(["player_name", "pitch_type"])
    .agg(
        pitches=("xRV_COL", "count"),
        xRV_COL=("xRV_COL", "mean"),
        xStuff_COL=("xStuff_COL", "mean"),
        avg_velo=("release_speed", "mean"),
        avg_ivb=("pfx_z", "mean"),
        avg_hb=("pfx_x", "mean"),
        avg_spin=("release_spin_rate", "mean"),
        avg_ext=("release_extension", "mean"),
        avg_vaa=("vaa", "mean"),
    )
    .reset_index()
    .query("pitches >= 50")
    .round(3)
)

print(f"\nPitcher-pitch type combos with 50+ Coors pitches: {len(pitcher_summary)}")
print(pitcher_summary.sort_values("xStuff_COL", ascending=False).head(10).to_string(index=False))


# ── 11. Export JSON for GitHub Pages ──────────────────────────────────────
os.makedirs("data", exist_ok=True)

# pitch-level export (sample for file size)
pitch_export_cols = [
    "player_name", "pitch_type", "game_date",
    "release_speed", "pfx_z", "pfx_x",
    "release_spin_rate", "release_extension",
    "vaa", "haa", "plate_x", "plate_z",
    "xRV_COL", "xStuff_COL",
]
pitch_export = coors_pred[
    [c for c in pitch_export_cols if c in coors_pred.columns]
].dropna().sample(min(5000, len(coors_pred)), random_state=99)

pitch_export.to_json(
    "data/coors_xgboost_pitches.json",
    orient="records", indent=2, date_format="iso"
)

# pitcher summary export
pitcher_summary.to_json(
    "data/coors_xgboost_pitchers.json",
    orient="records", indent=2
)

# feature importance export
importance_export = [
    {"feature": k, "importance": round(float(v), 4), "alpha": ALPHA.get(k, 1.0)}
    for k, v in sorted(importance.items(), key=lambda x: -x[1])
]
with open("data/coors_feature_importance.json", "w") as f:
    json.dump(importance_export, f, indent=2)

# model metadata
meta = {
    "train_r2":   round(train_r2, 4),
    "test_r2":    round(test_r2, 4),
    "test_rmse":  round(test_rmse, 4),
    "n_features": len(FEATURES),
    "features":   FEATURES,
    "models_used": ["tjStuff+ v2", "FanGraphs Stuff+/PitchingBot", "aStuff+ v2"],
    "coors_adjustments": ALPHA,
    "park_intercept": PARK_INTERCEPT,
    "hr_park_factor": HR_PARK_FACTOR,
    "babip_lift": BABIP_LIFT,
    "training_years": "2021-2024",
    "n_estimators": model.best_iteration,
}
with open("data/coors_model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nExported:")
print("  data/coors_xgboost_pitches.json")
print("  data/coors_xgboost_pitchers.json")
print("  data/coors_feature_importance.json")
print("  data/coors_model_meta.json")
print("\nDeploy with:")
print("  git add data/ && git commit -m 'add Coors XGBoost model output' && git push")
