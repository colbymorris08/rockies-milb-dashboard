# Rockies analytics dashboard (Coors Field pitching)

Static site for GitHub Pages: [rockies-milb-dashboard](https://colbymorris08.github.io/rockies-milb-dashboard/).

## Live pages

| Page | URL |
|------|-----|
| Movement simulator | [coors_movement_sim.html](https://colbymorris08.github.io/rockies-milb-dashboard/coors_movement_sim.html) |
| Top pitchers at Coors | [coors_top20.html](https://colbymorris08.github.io/rockies-milb-dashboard/coors_top20.html) |

## Coors XGBoost model (`coors_xgboost_model.py`)

Trains an **XGBoost** regressor on **Statcast 2021–2024** regular-season pitches (`pybaseball.statcast`), targeting **`estimated_woba_using_speedangle`** as a proxy for expected run value on the pitch.

**Feature philosophy (stuff signal):** inputs combine themes from three public “stuff” frameworks:

- **tjStuff+ v2 (Nestico):** velocity, movement, spin, extension, tunneling deltas vs. primary fastball.
- **FanGraphs Stuff+ / PitchingBot (Grove/Sarris):** release height/side, batter handedness, pitch type (encoded). **Plate location (`plate_x` / `plate_z`) is excluded** so the model does not learn pure location.
- **aStuff+ v2 (Salorio):** vertical/horizontal approach angles, spin axis.

**Altitude:** pitches at Coors (`home_team == COL`) get **physics-derived α multipliers** on selected movement/release features (documented in code) before training, reflecting reduced air density (~82% of sea level).

**Outputs (written to `data/`):**

- `coors_xgboost_pitchers.json` — pitcher × pitch type aggregates at Coors.
- `coors_xgboost_pitches.json` — sample of scored pitches (for exploration).
- `coors_feature_importance.json` — feature importances from the trained model.
- `coors_model_meta.json` — metrics and metadata.

Raw Statcast CSVs are **not** committed; they are pulled on demand via `pybaseball` (cached locally by pybaseball).

### Run the XGBoost model

```bash
cd ~/rockies
python3 -m pip install pybaseball xgboost scikit-learn pandas numpy
python3 coors_xgboost_model.py
```

The Statcast pull is large and may take a long time on first run.

---

## Two leaderboards (`generate_coors_stuff.py`)

This script trains **two** XGBoost models on the same target:

1. **Sea-level** — unadjusted Coors pitches combined with all other parks.
2. **Coors-adjusted** — Coors pitches get the same α multipliers as in `coors_xgboost_model.py`, then combined with other parks.

Each Coors pitch receives **Stuff+** on a **100 / 10** scale (matching the tjStuff+ convention): standardized model output per pitch, then aggregated.

It also writes **leaderboard JSON** for the site:

| File | Description |
|------|-------------|
| `data/coors_top20.json` | Top **20** pitchers at Coors among **all** teams — minimum **450** Coors pitches (~30 IP equivalent). Includes optional FanGraphs outcome merge for composite ranking. |
| `data/coors_pitchers_rockies.json` | Top **20** **Rockies** pitchers **pitching at Coors** (home half: `inning_topbot == Top` when `home_team == COL`) — minimum **150** pitches (~10 IP). |
| `data/coors_pitch_breakdown.json` | Per pitch-type **stuff_sl** and **stuff_col** for pitchers appearing on either leaderboard (30+ pitches per type). |
| `data/coors_importance_sl.json` / `data/coors_importance_col.json` | Feature importances for sea-level vs Coors-adjusted models. |
| `data/coors_model_meta.json` | Generation metadata, feature list, thresholds. |
| `data/coors_pitch_sample.json` | 5k pitch sample for tooling. |

### Run the generator

```bash
cd ~/rockies
python3 generate_coors_stuff.py
```

Then deploy:

```bash
git add data/ coors_xgboost_model.py generate_coors_stuff.py coors_top20.html coors_movement_sim.html README.md
git commit -m "complete Coors XGBoost model — pure stuff signal, two leaderboards, full site integration"
git push
```

---

## Frontend notes

- **`coors_top20.html`** loads `data/coors_top20.json` and `data/coors_pitchers_rockies.json` and switches between **All at Coors** and **Rockies at Coors** tabs.
- **`coors_movement_sim.html`** loads `data/coors_pitch_breakdown.json` for the Stuff+ panel: **side-by-side sea-level (SL) vs Coors-adjusted (COL)** bars per pitch type for the selected pitcher (names matched via `Last, First` → `First Last`).

---

## Other assets

- `col_minors_dashboard.html` — minors dashboard (separate data pipeline).
- `update_dashboard.sh` — helper to refresh minors cache (Playwright).
