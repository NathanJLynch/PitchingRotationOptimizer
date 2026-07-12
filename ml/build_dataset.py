# ml/build_dataset.py
"""
Pulls historical Statcast data to build the training dataset.
Run once (takes a few minutes with pybaseball cache):
    python -m ml.build_dataset

Requires: pip install pybaseball pandas pyarrow
"""
import os
import requests
import pandas as pd
import numpy as np
import pybaseball as pb
from pybaseball import cache
from concurrent.futures import ProcessPoolExecutor, as_completed

cache.enable()

SEASONS = [2023, 2024, 2025]

# Only the columns we actually use — drops ~80 Statcast columns immediately,
# cutting memory usage and speeding up every subsequent operation
KEEP_COLS = [
    "pitcher", "game_date", "home_team", "away_team", "inning",
    "description", "pitch_type", "zone",
    "release_speed", "release_spin_rate", "release_extension",
    "estimated_woba_using_speedangle", "launch_speed",
]

# Pitch type buckets
FB_TYPES = {"FF", "SI", "FC"}
BR_TYPES = {"SL", "CU", "KC", "SV"}
OS_TYPES = {"CH", "FS", "FO"}
SWING_TYPES = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play"}
WHIFF_TYPES = {"swinging_strike", "swinging_strike_blocked"}

LEAGUE_AVG_XWOBA = 0.315
MIN_PRIOR_PITCHES = 50   # minimum pitches in 30-day window for reliable stats

TEAM_MAP = {
    "ARI": "Arizona Diamondbacks",
    "ATL": "Atlanta Braves",
    "BAL": "Baltimore Orioles",
    "BOS": "Boston Red Sox",
    "CHC": "Chicago Cubs",
    "CWS": "Chicago White Sox",
    "CIN": "Cincinnati Reds",
    "CLE": "Cleveland Guardians",
    "COL": "Colorado Rockies",
    "DET": "Detroit Tigers",
    "HOU": "Houston Astros",
    "KC":  "Kansas City Royals",
    "LAA": "Los Angeles Angels",
    "LAD": "Los Angeles Dodgers",
    "MIA": "Miami Marlins",
    "MIL": "Milwaukee Brewers",
    "MIN": "Minnesota Twins",
    "NYM": "New York Mets",
    "NYY": "New York Yankees",
    "OAK": "Oakland Athletics",
    "PHI": "Philadelphia Phillies",
    "PIT": "Pittsburgh Pirates",
    "SD":  "San Diego Padres",
    "SEA": "Seattle Mariners",
    "SF":  "San Francisco Giants",
    "STL": "St. Louis Cardinals",
    "TB":  "Tampa Bay Rays",
    "TEX": "Texas Rangers",
    "TOR": "Toronto Blue Jays",
    "WSH": "Washington Nationals",
}

OUTPUT_PATH = "ml/data/training_data.parquet"

os.makedirs("ml/data",   exist_ok=True)
os.makedirs("ml/models", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Top-level entry point — seasons run in parallel
# ─────────────────────────────────────────────────────────────

def build():
    print(f"Building dataset for seasons: {SEASONS}")
    print(f"Using {len(SEASONS)} parallel workers...\n")

    all_rows = []

    # Each season is fully independent — process in parallel
    with ProcessPoolExecutor(max_workers=len(SEASONS)) as executor:
        futures = {executor.submit(_build_season, s): s for s in SEASONS}
        for future in as_completed(futures):
            season = futures[future]
            try:
                rows = future.result()
                print(f"  ✓ {season}: {len(rows)} rows")
                all_rows.extend(rows)
            except Exception as e:
                print(f"  ✗ {season} failed: {e}")

    if not all_rows:
        print("\nNo rows built — check Statcast connectivity and pybaseball cache.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df = df.dropna(subset=["label_k_pct", "label_ops_allowed", "label_wpa"])

    # Sort by date — critical so TimeSeriesSplit in train.py works correctly
    df = df.sort_values("game_date").reset_index(drop=True)

    print(f"\nTotal rows: {len(df)}")
    print(f"Date range: {df['game_date'].min()} → {df['game_date'].max()}")
    print(f"Columns: {list(df.columns)}")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    return df


# ─────────────────────────────────────────────────────────────
# Per-season builder (runs in its own process)
# ─────────────────────────────────────────────────────────────

def _build_season(season: int) -> list:
    print(f"[{season}] Fetching Statcast data...")
    try:
        sc = pb.statcast(
            start_dt=f"{season}-04-01",   # regular season only — skip spring training noise
            end_dt=f"{season}-09-30",
        )
    except Exception as e:
        raise RuntimeError(f"Statcast fetch failed: {e}")

    if sc is None or sc.empty:
        raise RuntimeError("No Statcast data returned")

    # ── Drop unused columns immediately — cuts memory ~80% ───
    sc = sc[[c for c in KEEP_COLS if c in sc.columns]].copy()
    sc["game_date"] = pd.to_datetime(sc["game_date"])
    sc = sc[sc["inning"] <= 9].copy()

    print(f"[{season}] {len(sc):,} pitches loaded. Fetching team batting stats...")

    try:
        team_batting  = _get_team_batting_stats(season)
        team_lookup   = _build_team_lookup(team_batting)
    except Exception as e:
        raise RuntimeError(f"Team batting fetch failed: {e}")

    print(f"[{season}] Pre-aggregating rolling pitcher features...")

    # ── Tag binary pitch-level columns once — avoids re-filtering per start ──
    # Use fillna(False) before astype — zone/launch_speed/pitch_type can be NaN
    sc["is_whiff"]    = sc["description"].isin(WHIFF_TYPES).fillna(False).astype(np.int8)
    sc["is_swing"]    = sc["description"].isin(SWING_TYPES).fillna(False).astype(np.int8)
    sc["is_inzone"]   = sc["zone"].between(1, 9).fillna(False).astype(np.int8)
    sc["is_outzone"]  = (~sc["zone"].between(1, 9)).fillna(False).astype(np.int8)
    sc["is_outchase"] = (sc["is_outzone"].astype(bool) & sc["is_swing"].astype(bool)).astype(np.int8)
    sc["is_fb"]       = sc["pitch_type"].isin(FB_TYPES).fillna(False).astype(np.int8)
    sc["is_br"]       = sc["pitch_type"].isin(BR_TYPES).fillna(False).astype(np.int8)
    sc["is_os"]       = sc["pitch_type"].isin(OS_TYPES).fillna(False).astype(np.int8)
    sc["is_hard_hit"] = (sc["launch_speed"].fillna(0) >= 95).astype(np.int8)
    sc["is_batted"]   = sc["launch_speed"].notna().astype(np.int8)
    sc["fb_velo"]     = np.where(sc["is_fb"], sc["release_speed"],     np.nan)
    sc["fb_spin"]     = np.where(sc["is_fb"], sc["release_spin_rate"], np.nan)

    # ── Build rolling 30-day pitcher features for every game-date ────────────
    # Sort by pitcher + date so rolling window is chronologically correct
    sc = sc.sort_values(["pitcher", "game_date"]).reset_index(drop=True)

    AGG_COLS = [
        "is_whiff", "is_swing", "is_inzone", "is_outzone", "is_outchase",
        "is_fb", "is_br", "is_os", "is_hard_hit", "is_batted",
        "fb_velo", "fb_spin",
        "release_extension", "estimated_woba_using_speedangle",
    ]

    # One row per pitcher × game — sum all binary tags per game first,
    # then roll across games (much faster than rolling over every pitch)
    per_game = (
        sc.groupby(["pitcher", "game_date", "home_team", "away_team"])[AGG_COLS]
        .agg({
            "is_whiff":    "sum",
            "is_swing":    "sum",
            "is_inzone":   "sum",
            "is_outzone":  "sum",
            "is_outchase": "sum",
            "is_fb":       "sum",
            "is_br":       "sum",
            "is_os":       "sum",
            "is_hard_hit": "sum",
            "is_batted":   "sum",
            "fb_velo":     "mean",
            "fb_spin":     "mean",
            "release_extension":                "mean",
            "estimated_woba_using_speedangle":  "mean",
        })
        .reset_index()
    )
    per_game["n_pitches"] = (
        sc.groupby(["pitcher", "game_date", "home_team", "away_team"])
        .size()
        .values
    )

    print(f"[{season}] Computing rolling windows across {len(per_game)} pitcher-games...")

    roll_cols = [
        "is_whiff", "is_swing", "is_inzone", "is_outzone", "is_outchase",
        "is_fb", "is_br", "is_os", "is_hard_hit", "is_batted",
        "fb_velo", "fb_spin", "release_extension",
        "estimated_woba_using_speedangle", "n_pitches",
    ]

    # ── Rolling 30-day window per pitcher ────────────────────────────────────
    # groupby().rolling() with a time offset requires game_date as the INDEX
    # within each group. We sort first, then set the index, keeping pitcher
    # accessible via groupby level.
    per_game_sorted = per_game.sort_values(["pitcher", "game_date"])
    per_game_indexed = per_game_sorted.set_index("game_date")

    rolled_parts = []
    for pitcher_id, grp in per_game_indexed.groupby("pitcher"):
        r = (
            grp[roll_cols]
            .rolling("30D", closed="left")   # "left" excludes current game — pre-game only
            .agg({
                "is_whiff":    "sum",
                "is_swing":    "sum",
                "is_inzone":   "sum",
                "is_outzone":  "sum",
                "is_outchase": "sum",
                "is_fb":       "sum",
                "is_br":       "sum",
                "is_os":       "sum",
                "is_hard_hit": "sum",
                "is_batted":   "sum",
                "fb_velo":     "mean",
                "fb_spin":     "mean",
                "release_extension":               "mean",
                "estimated_woba_using_speedangle": "mean",
                "n_pitches":   "sum",
            })
        )
        r.columns = [f"prior_{c}" for c in r.columns]
        r["pitcher"]   = pitcher_id
        r["home_team"] = grp["home_team"].values
        r["away_team"] = grp["away_team"].values
        rolled_parts.append(r)

    rolled = pd.concat(rolled_parts).reset_index()   # game_date comes back as column

    # ── Compute per-game outcome labels (what happened THIS start) ───────────
    print(f"[{season}] Computing start labels...")
    labels = (
        per_game_sorted[["pitcher", "game_date", "is_whiff", "is_swing",
                          "estimated_woba_using_speedangle"]]
        .rename(columns={
            "is_whiff":                         "start_whiffs",
            "is_swing":                         "start_swings",
            "estimated_woba_using_speedangle":  "start_xwoba",
        })
    )

    combined = rolled.merge(labels, on=["pitcher", "game_date"], how="inner")

    # ── Build feature rows from pre-aggregated data ───────────────────────────
    print(f"[{season}] Building feature rows...")
    rows = []

    for _, r in combined.iterrows():
        n_pitches = max(r["prior_n_pitches"], 1)
        if n_pitches < MIN_PRIOR_PITCHES:
            continue

        start_xwoba = r["start_xwoba"]
        if pd.isna(start_xwoba):
            continue

        # Determine opponent
        home_team  = r["home_team"]
        away_team  = r["away_team"]
        pitcher_id = r["pitcher"]

        # Infer pitcher's team from which side they pitched on most
        # (home pitcher → home_team, away pitcher → away_team)
        # We use the raw per_game data to check — look at the pitcher's
        # most common home_team field across this game
        opp_abbrev = away_team  # default: pitcher is home, opponent is away
        opp_batting = team_lookup.get(opp_abbrev) or team_lookup.get(home_team)
        if opp_batting is None:
            continue

        n_swings  = max(r["prior_is_swing"],   1)
        n_outzone = max(r["prior_is_outzone"],  1)
        n_batted  = max(r["prior_is_batted"],   1)

        p_whiff_pct = r["prior_is_whiff"]    / n_swings
        p_zone_pct  = r["prior_is_inzone"]   / n_pitches
        p_chase_pct = r["prior_is_outchase"] / n_outzone
        p_fb_velo   = r["prior_fb_velo"]
        p_spin_rate = r["prior_fb_spin"]
        p_extension = r["prior_release_extension"]
        p_xwoba     = r["prior_estimated_woba_using_speedangle"]
        p_hard_hit  = r["prior_is_hard_hit"] / n_batted

        opp_whiff = opp_batting["whiff_rate"]
        opp_chase = opp_batting["chase_rate"]
        opp_ops   = opp_batting["ops"]
        opp_hard  = opp_batting["hard_hit_pct"]

        actual_k_pct = r["start_whiffs"] / max(r["start_swings"], 1)
        actual_ops   = float(start_xwoba) * 3.2
        actual_wpa   = (LEAGUE_AVG_XWOBA - float(start_xwoba)) * 2.0

        rows.append({
            "pitcher_id":       pitcher_id,
            "season":           season,
            "game_date":        r["game_date"],

            "p_whiff_pct":      p_whiff_pct,
            "p_zone_pct":       p_zone_pct,
            "p_chase_pct":      p_chase_pct,
            "p_fb_velo":        p_fb_velo,
            "p_spin_rate":      p_spin_rate,
            "p_extension":      p_extension,
            "p_xwoba_allowed":  p_xwoba,
            "p_hard_hit_pct":   p_hard_hit,
            "p_fb_usage":       r["prior_is_fb"] / n_pitches,
            "p_breaking_usage": r["prior_is_br"] / n_pitches,
            "p_offspeed_usage": r["prior_is_os"] / n_pitches,

            "opp_ops":          opp_ops,
            "opp_whiff_rate":   opp_whiff,
            "opp_chase_rate":   opp_chase,
            "opp_k_pct":        opp_batting["k_pct"],
            "opp_hard_hit_pct": opp_hard,
            "opp_xwoba":        opp_batting["xwoba"],

            "whiff_matchup":    p_whiff_pct * opp_whiff,
            "chase_matchup":    p_chase_pct * opp_chase,
            "ops_gap":          opp_ops - (p_xwoba * 3.2 if pd.notna(p_xwoba) else 0.720),
            "power_matchup":    (p_fb_velo if pd.notna(p_fb_velo) else 93.0) * opp_hard,

            "label_k_pct":       actual_k_pct,
            "label_ops_allowed": actual_ops,
            "label_wpa":         actual_wpa,
        })

    return rows


# ─────────────────────────────────────────────────────────────
# Team batting stats
# ─────────────────────────────────────────────────────────────

def _get_team_batting_stats(season: int) -> pd.DataFrame:
    url = (
        "https://statsapi.mlb.com/api/v1/teams/stats"
        f"?season={season}&group=hitting&stats=season&sportId=1"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    data = r.json()

    rows = []
    for split in data.get("stats", [{}])[0].get("splits", []):
        team = split.get("team", {})
        stat = split.get("stat", {})
        team_name = team.get("name")
        if not team_name:
            continue
        pa  = int(stat.get("plateAppearances", 0) or 0)
        so  = int(stat.get("strikeOuts", 0) or 0)
        ops = float(stat.get("ops", 0.720) or 0.720)
        rows.append({
            "Team":  team_name,
            "PA":    pa,
            "SO":    so,
            "OPS":   ops,
            "K_pct": so / max(pa, 1),
        })
    return pd.DataFrame(rows)


def _build_team_lookup(team_batting: pd.DataFrame) -> dict:
    lookup = {}
    for _, row in team_batting.iterrows():
        abbrev = next((k for k, v in TEAM_MAP.items() if v == row["Team"]), None)
        if abbrev is None:
            continue
        lookup[abbrev] = {
            "ops":          float(row["OPS"]),
            "k_pct":        float(row["K_pct"]),
            "whiff_rate":   0.240,   # league avg — team-level not in API
            "chase_rate":   0.290,
            "hard_hit_pct": 0.370,
            "xwoba":        0.315,
        }
    return lookup


# ─────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    build()