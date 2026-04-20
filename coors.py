"""
Coors Field BABIP Analysis by Hit Type
+ Automated Free Agent K%/BB% Conclusions

Requirements:
    pip install pybaseball pandas numpy matplotlib seaborn requests

Usage:
    python coors_babip_analysis.py

Data Sources:
    - pybaseball (Baseball Savant Statcast) for BABIP / hit type
    - MLB Stats API for free agent K%/BB% rates
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import requests
import warnings
from pybaseball import statcast
from pybaseball import cache

warnings.filterwarnings("ignore")
cache.enable()  # Cache Statcast pulls so you don't re-download

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SEASONS = ["2022", "2023", "2024"]          # Years to analyze
COORS_TEAM = "COL"                           # Rockies = Coors home games
MIN_BIP = 50                                 # Minimum BIP for free agent filter
FREE_AGENT_SEASON = 2024                     # Season for FA discipline stats
OUTPUT_PLOT = "coors_babip_analysis.png"

# Hit type mapping from Statcast bb_type column
HIT_TYPE_LABELS = {
    "ground_ball": "Ground Ball",
    "line_drive":  "Line Drive",
    "popup":       "Pop Up",
    "fly_ball":    "Fly Ball",
}

# ─────────────────────────────────────────────
# STEP 1: PULL STATCAST DATA
# ─────────────────────────────────────────────
def pull_statcast_seasons(seasons: list[str]) -> pd.DataFrame:
    """Pull full season Statcast data for given years."""
    frames = []
    for year in seasons:
        print(f"  Pulling {year} Statcast data (this may take a few minutes)...")
        df = statcast(f"{year}-04-01", f"{year}-10-01")
        df["season"] = int(year)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────
# STEP 2: COMPUTE BABIP BY PARK & HIT TYPE
# ─────────────────────────────────────────────
def compute_babip(df: pd.DataFrame) -> dict:
    """
    BABIP = H / (AB - K - HR + SF)
    Using Statcast: filter balls in play, exclude HR and K,
    then check if event is a hit.
    """
    # Balls in play: exclude strikeouts, HRs, walks, HBP, catcher interference
    bip_events = {
        "single", "double", "triple", "home_run",
        "field_out", "grounded_into_double_play", "double_play",
        "force_out", "sac_fly", "field_error",
        "fielders_choice", "fielders_choice_out",
        "triple_play", "sac_fly_double_play"
    }
    hits = {"single", "double", "triple"}  # Exclude HR per BABIP definition

    bip = df[df["events"].isin(bip_events)].copy()
    bip = bip[bip["bb_type"].isin(HIT_TYPE_LABELS.keys())].copy()
    bip["is_hit"] = bip["events"].isin(hits).astype(int)
    bip["is_coors"] = (bip["home_team"] == COORS_TEAM).astype(int)
    bip["park_label"] = np.where(bip["is_coors"] == 1, "Coors Field", "All Other Parks")

    results = {}
    for hit_type, label in HIT_TYPE_LABELS.items():
        subset = bip[bip["bb_type"] == hit_type]
        grouped = (
            subset.groupby("park_label")["is_hit"]
            .agg(["sum", "count"])
            .rename(columns={"sum": "hits", "count": "bip"})
        )
        grouped["babip"] = grouped["hits"] / grouped["bip"]
        results[label] = grouped

    # Overall BABIP (all hit types combined)
    overall = (
        bip.groupby("park_label")["is_hit"]
        .agg(["sum", "count"])
        .rename(columns={"sum": "hits", "count": "bip"})
    )
    overall["babip"] = overall["hits"] / overall["bip"]
    results["Overall"] = overall

    return results, bip


# ─────────────────────────────────────────────
# STEP 3: PULL FREE AGENT DISCIPLINE STATS
# ─────────────────────────────────────────────
def get_free_agent_discipline(season: int) -> pd.DataFrame:
    """
    Pull K% and BB% from MLB Stats API (no key needed).
    Returns qualified hitters with discipline metrics.
    """
    print(f"\n  Fetching {season} hitter stats from MLB Stats API...")
    url = (
        f"https://statsapi.mlb.com/api/v1/stats"
        f"?stats=season&season={season}&group=hitting"
        f"&gameType=R&limit=500&sortStat=plateAppearances"
    )
    r = requests.get(url, timeout=20)
    data = r.json()

    rows = []
    for entry in data.get("stats", [{}])[0].get("splits", []):
        s = entry.get("stat", {})
        p = entry.get("player", {})
        rows.append({
            "name":         p.get("fullName"),
            "player_id":    p.get("id"),
            "pa":           int(s.get("plateAppearances", 0)),
            "ab":           int(s.get("atBats", 0)),
            "hits":         int(s.get("hits", 0)),
            "hr":           int(s.get("homeRuns", 0)),
            "bb":           int(s.get("baseOnBalls", 0)),
            "so":           int(s.get("strikeOuts", 0)),
            "avg":          float(s.get("avg", 0) or 0),
            "obp":          float(s.get("obp", 0) or 0),
            "slg":          float(s.get("slg", 0) or 0),
        })

    df = pd.DataFrame(rows)
    df = df[df["pa"] >= MIN_BIP].copy()
    df["k_pct"] = df["so"] / df["pa"]
    df["bb_pct"] = df["bb"] / df["pa"]
    df["bb_k_ratio"] = df["bb"] / df["so"].replace(0, np.nan)
    df["ops"] = df["obp"] + df["slg"]
    return df


# ─────────────────────────────────────────────
# STEP 4: GENERATE AUTOMATED CONCLUSIONS
# ─────────────────────────────────────────────
def generate_conclusions(babip_results: dict, fa_df: pd.DataFrame, bip_df: pd.DataFrame) -> str:
    """
    Auto-generate analytical conclusions based on:
    - Coors BABIP inflation by hit type
    - Free agent K% / BB% profiles that suggest Coors risk/opportunity
    """
    lines = []
    lines.append("=" * 70)
    lines.append("  AUTOMATED ANALYSIS: COORS BABIP + FREE AGENT TARGETING")
    lines.append("=" * 70)

    # ── Part 1: BABIP inflation by hit type ──
    lines.append("\n📊 COORS FIELD BABIP INFLATION BY HIT TYPE")
    lines.append("-" * 50)

    inflations = {}
    for hit_type, tbl in babip_results.items():
        if "Coors Field" not in tbl.index or "All Other Parks" not in tbl.index:
            continue
        coors_val  = tbl.loc["Coors Field",    "babip"]
        others_val = tbl.loc["All Other Parks", "babip"]
        inflation  = coors_val - others_val
        inflations[hit_type] = inflation

        direction = "HIGHER" if inflation > 0 else "LOWER"
        lines.append(
            f"  {hit_type:<15}: Coors={coors_val:.3f} | Elsewhere={others_val:.3f} "
            f"| Δ={inflation:+.3f} ({direction})"
        )

    # Most / least inflated
    if inflations:
        most_inflated  = max(inflations, key=lambda x: inflations[x])
        least_inflated = min(inflations, key=lambda x: inflations[x])
        lines.append(
            f"\n  → Most Coors-inflated hit type: {most_inflated} "
            f"(+{inflations[most_inflated]:.3f} BABIP)"
        )
        lines.append(
            f"  → Least inflated / deflated:    {least_inflated} "
            f"({inflations[least_inflated]:+.3f} BABIP)"
        )

    # ── Part 2: FA targeting conclusions ──
    lines.append("\n\n🎯 FREE AGENT HITTER TARGETING — K% / BB% ANALYSIS")
    lines.append("-" * 50)

    # Coors-friendly profile: low K%, decent BB%, high GB% helps (less inflation there)
    # BABIP inflation is largest on fly balls and line drives typically
    # Classify hitters
    k_threshold  = fa_df["k_pct"].quantile(0.35)   # Bottom 35% K = low strikeout
    bb_threshold = fa_df["bb_pct"].quantile(0.60)  # Top 40% BB = disciplined

    low_k  = fa_df[fa_df["k_pct"]  <= k_threshold]
    high_bb = fa_df[fa_df["bb_pct"] >= bb_threshold]
    elite_contact = fa_df[
        (fa_df["k_pct"]  <= k_threshold) &
        (fa_df["bb_pct"] >= bb_threshold)
    ].sort_values("ops", ascending=False)

    lines.append(
        f"\n  Discipline thresholds used:"
        f"\n    Low K%  ≤ {k_threshold:.1%}  (bottom 35th percentile)"
        f"\n    High BB% ≥ {bb_threshold:.1%} (top 40th percentile)"
    )

    # Key insight: which hit types are most inflated?
    gb_inflation = inflations.get("Ground Ball", 0)
    fb_inflation = inflations.get("Fly Ball", 0)
    ld_inflation = inflations.get("Line Drive", 0)

    lines.append("\n  📌 KEY FINDINGS:")

    if fb_inflation > gb_inflation:
        lines.append(
            f"\n  • Fly ball inflation ({fb_inflation:+.3f}) exceeds ground ball inflation "
            f"({gb_inflation:+.3f}) at Coors. Power hitters who elevate are PARTICULARLY "
            f"favored — their BABIP inflates disproportionately vs. the rest of MLB."
        )
    else:
        lines.append(
            f"\n  • Ground ball inflation ({gb_inflation:+.3f}) leads at Coors, which is unusual. "
            f"Gap hitters and spray hitters benefit most."
        )

    if ld_inflation > 0.02:
        lines.append(
            f"\n  • Line drive BABIP is significantly inflated (+{ld_inflation:.3f}). "
            f"Contact-first hitters who barrel frequently are a strong target — "
            f"their natural skill plays up further at altitude."
        )

    lines.append(
        f"\n  • Low-K hitters (K% ≤ {k_threshold:.1%}) are safer bets: "
        f"they put more balls in play, generating more opportunities for Coors BABIP inflation. "
        f"With {len(low_k)} qualifying hitters in this tier, the pool is workable."
    )

    lines.append(
        f"\n  • High-BB hitters (BB% ≥ {bb_threshold:.1%}) (n={len(high_bb)}) bring a double "
        f"benefit: discipline reduces K-driven outs AND generates free bases. At Coors, "
        f"where runners score at elevated rates, this compounds value."
    )

    if not elite_contact.empty:
        lines.append(f"\n  ⭐ ELITE CONTACT + DISCIPLINE (Low K% AND High BB%) — Top 10 by OPS:")
        for _, row in elite_contact.head(10).iterrows():
            lines.append(
                f"    {row['name']:<25} | K%={row['k_pct']:.1%} | BB%={row['bb_pct']:.1%} "
                f"| OPS={row['ops']:.3f} | BB/K={row['bb_k_ratio']:.2f}"
            )

    # Hitters to AVOID at Coors (high K, low BB — can't take advantage of BABIP inflation)
    risky = fa_df[
        (fa_df["k_pct"]  > fa_df["k_pct"].quantile(0.70)) &
        (fa_df["bb_pct"] < fa_df["bb_pct"].quantile(0.30))
    ].sort_values("k_pct", ascending=False)

    lines.append(f"\n  ⚠️  HIGH-K / LOW-BB HITTERS — Risky Coors Targets (Top 10 by K%):")
    for _, row in risky.head(10).iterrows():
        lines.append(
            f"    {row['name']:<25} | K%={row['k_pct']:.1%} | BB%={row['bb_pct']:.1%} "
            f"| OPS={row['ops']:.3f}"
        )

    lines.append(
        "\n  → These hitters have fewer BIP events, meaning they benefit less from "
        "Coors BABIP inflation. High K rates don't get better at altitude — "
        "and teams overpay for power that strikeout-heavy hitters don't fully deliver."
    )

    # ── Part 3: Overall recommendation ──
    overall_coors  = babip_results.get("Overall", pd.DataFrame())
    if "Coors Field" in overall_coors.index and "All Other Parks" in overall_coors.index:
        coors_overall  = overall_coors.loc["Coors Field",    "babip"]
        others_overall = overall_coors.loc["All Other Parks", "babip"]
        overall_delta  = coors_overall - others_overall
        lines.append(f"\n\n📋 SUMMARY RECOMMENDATION")
        lines.append("-" * 50)
        lines.append(
            f"  Overall BABIP at Coors ({coors_overall:.3f}) runs "
            f"{overall_delta:+.3f} vs. the rest of MLB ({others_overall:.3f}). "
            f"That's a meaningful edge baked into every ball in play."
        )
        lines.append(
            f"\n  For a Rockies front office targeting free agent hitters, the optimal profile is:"
            f"\n    1. K% ≤ {k_threshold:.1%}  → maximizes balls in play to exploit BABIP inflation"
            f"\n    2. BB% ≥ {bb_threshold:.1%} → on-base floor + run-scoring leverage at altitude"
            f"\n    3. Prefer fly ball / line drive tendencies given their larger Coors BABIP boost"
            f"\n    4. Avoid pure power-or-nothing profiles (HR/FB + high K) — "
            f"the BABIP edge is wasted on hitters who miss too often"
        )

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ─────────────────────────────────────────────
# STEP 5: VISUALIZATION
# ─────────────────────────────────────────────
def plot_results(babip_results: dict, fa_df: pd.DataFrame, output_path: str):
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0d0d1a")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    purple   = "#9B59B6"
    gold     = "#F4D03F"
    teal     = "#1ABC9C"
    red      = "#E74C3C"
    bg       = "#0d0d1a"
    card_bg  = "#16213e"
    text_col = "#ECF0F1"

    # ── Plot 1: BABIP by hit type (grouped bar) ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor(card_bg)

    hit_types = [k for k in HIT_TYPE_LABELS.values()] + ["Overall"]
    coors_vals  = []
    others_vals = []

    for label in hit_types:
        tbl = babip_results.get(label, pd.DataFrame())
        coors_vals.append(
            tbl.loc["Coors Field",    "babip"] if "Coors Field"    in tbl.index else np.nan
        )
        others_vals.append(
            tbl.loc["All Other Parks", "babip"] if "All Other Parks" in tbl.index else np.nan
        )

    x = np.arange(len(hit_types))
    w = 0.35
    bars1 = ax1.bar(x - w/2, coors_vals,  w, label="Coors Field",    color=purple, alpha=0.9, zorder=3)
    bars2 = ax1.bar(x + w/2, others_vals, w, label="All Other Parks", color=teal,   alpha=0.9, zorder=3)

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{bar.get_height():.3f}", ha="center", va="bottom",
                 fontsize=9, color=text_col, fontweight="bold")
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                 f"{bar.get_height():.3f}", ha="center", va="bottom",
                 fontsize=9, color=text_col, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(hit_types, color=text_col, fontsize=11)
    ax1.set_ylabel("BABIP", color=text_col, fontsize=11)
    ax1.set_title("BABIP by Hit Type: Coors Field vs. All Other Parks",
                  color=gold, fontsize=14, fontweight="bold", pad=12)
    ax1.legend(facecolor=card_bg, labelcolor=text_col, fontsize=10)
    ax1.tick_params(colors=text_col)
    ax1.spines[:].set_color("#333355")
    ax1.set_ylim(0, max([v for v in coors_vals + others_vals if not np.isnan(v)]) * 1.15)
    ax1.yaxis.label.set_color(text_col)
    ax1.grid(axis="y", alpha=0.2, color="#444466", zorder=0)

    # ── Plot 2: BABIP inflation (delta) ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(card_bg)

    deltas = [c - o if not (np.isnan(c) or np.isnan(o)) else 0
              for c, o in zip(coors_vals, others_vals)]
    colors_delta = [red if d < 0 else purple for d in deltas]
    bars = ax2.barh(hit_types, deltas, color=colors_delta, alpha=0.88, zorder=3)

    for bar, val in zip(bars, deltas):
        ax2.text(val + (0.001 if val >= 0 else -0.001),
                 bar.get_y() + bar.get_height()/2,
                 f"{val:+.3f}", va="center",
                 ha="left" if val >= 0 else "right",
                 color=text_col, fontsize=9, fontweight="bold")

    ax2.axvline(0, color="#aaaacc", linewidth=0.8)
    ax2.set_xlabel("BABIP Delta (Coors − Others)", color=text_col, fontsize=10)
    ax2.set_title("Coors BABIP Inflation\nby Hit Type", color=gold,
                  fontsize=12, fontweight="bold")
    ax2.tick_params(colors=text_col)
    ax2.spines[:].set_color("#333355")
    ax2.xaxis.label.set_color(text_col)
    ax2.grid(axis="x", alpha=0.2, color="#444466", zorder=0)
    ax2.set_yticklabels(hit_types, color=text_col)

    # ── Plot 3: FA K% vs BB% scatter ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(card_bg)

    k_thresh  = fa_df["k_pct"].quantile(0.35)
    bb_thresh = fa_df["bb_pct"].quantile(0.60)

    scatter = ax3.scatter(
        fa_df["k_pct"], fa_df["bb_pct"],
        c=fa_df["ops"], cmap="RdYlGn",
        s=fa_df["pa"] / 10, alpha=0.65, zorder=3
    )
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label("OPS", color=text_col)
    cbar.ax.yaxis.set_tick_params(color=text_col)
    plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=text_col)

    ax3.axvline(k_thresh,  color=red,  linestyle="--", linewidth=1.2, alpha=0.7, label=f"Low K% ≤{k_thresh:.1%}")
    ax3.axhline(bb_thresh, color=teal, linestyle="--", linewidth=1.2, alpha=0.7, label=f"High BB% ≥{bb_thresh:.1%}")

    # Label top-right quadrant (ideal Coors targets)
    ideal = fa_df[(fa_df["k_pct"] <= k_thresh) & (fa_df["bb_pct"] >= bb_thresh)]
    for _, row in ideal.nlargest(5, "ops").iterrows():
        ax3.annotate(row["name"].split()[-1],
                     (row["k_pct"], row["bb_pct"]),
                     fontsize=7, color=gold, alpha=0.9,
                     xytext=(4, 4), textcoords="offset points")

    ax3.set_xlabel("K%", color=text_col, fontsize=10)
    ax3.set_ylabel("BB%", color=text_col, fontsize=10)
    ax3.set_title("FA Hitters: K% vs BB%\n(Size = PA, Color = OPS)",
                  color=gold, fontsize=12, fontweight="bold")
    ax3.legend(facecolor=card_bg, labelcolor=text_col, fontsize=8, loc="upper right")
    ax3.tick_params(colors=text_col)
    ax3.spines[:].set_color("#333355")
    ax3.xaxis.label.set_color(text_col)
    ax3.yaxis.label.set_color(text_col)
    ax3.grid(alpha=0.15, color="#444466", zorder=0)
    ax3.invert_xaxis()  # Lower K% = better → on right side

    fig.suptitle(
        "Coors Field BABIP Analysis & Free Agent Targeting Tool",
        color=gold, fontsize=16, fontweight="bold", y=0.98
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=bg)
    print(f"\n  Chart saved → {output_path}")
    plt.show()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  COORS FIELD BABIP ANALYSIS + FREE AGENT TARGETING")
    print("=" * 70)

    # 1. Pull Statcast
    print(f"\n[1/4] Pulling Statcast data for {SEASONS}...")
    raw = pull_statcast_seasons(SEASONS)
    print(f"  → {len(raw):,} pitches pulled")

    # 2. BABIP by park + hit type
    print("\n[2/4] Computing BABIP by park and hit type...")
    babip_results, bip_df = compute_babip(raw)
    for label, tbl in babip_results.items():
        print(f"\n  {label}:")
        print(tbl[["hits", "bip", "babip"]].to_string())

    # 3. Free agent discipline stats
    print("\n[3/4] Fetching free agent discipline stats...")
    fa_df = get_free_agent_discipline(FREE_AGENT_SEASON)
    print(f"  → {len(fa_df)} qualified hitters loaded")

    # 4. Generate conclusions
    print("\n[4/4] Generating automated conclusions...")
    conclusions = generate_conclusions(babip_results, fa_df, bip_df)
    print(conclusions)

    # 5. Plot
    print("\n[Plotting]...")
    plot_results(babip_results, fa_df, OUTPUT_PLOT)

    # Save conclusions to text file
    with open("coors_babip_conclusions.txt", "w") as f:
        f.write(conclusions)
    print("\n  Text conclusions saved → coors_babip_conclusions.txt")


if __name__ == "__main__":
    main()
