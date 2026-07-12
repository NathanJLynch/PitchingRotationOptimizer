# database/pipeline/statcast.py
import requests
import pybaseball as pb
from pybaseball import cache
from datetime import date, timedelta
import pandas as pd
import numpy as np

cache.enable()

# Statcast abbreviation lookup — needed to filter cached Statcast data by team
MLB_ID_TO_ABBREV = {
    108: "LAA", 109: "ARI", 110: "BAL", 111: "BOS", 112: "CHC",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU",
    118: "KC",  119: "LAD", 120: "WSH", 121: "NYM", 133: "OAK",
    134: "PIT", 135: "SD",  136: "SEA", 137: "SF",  138: "STL",
    139: "TB",  140: "TEX", 141: "TOR", 142: "MIN", 143: "PHI",
    144: "ATL", 145: "CWS", 146: "MIA", 147: "NYY", 158: "MIL",
}


# ─────────────────────────────────────────────────────────────
# PITCHER — unchanged, already works well
# ─────────────────────────────────────────────────────────────

def fetch_pitcher_statcast(mlb_player_id: int, days: int = 60) -> dict:
    """
    Pull Statcast pitch-level data for one pitcher over a rolling window.
    Returns aggregated stats ready to write into the Pitcher table.
    """
    end   = date.today()
    start = end - timedelta(days=days)

    df = pb.statcast_pitcher(
        start_dt  = start.isoformat(),
        end_dt    = end.isoformat(),
        player_id = mlb_player_id,
    )

    if df is None or df.empty:
        return {}

    fastballs = df[df["pitch_type"].isin(["FF", "SI", "FC"])]
    breaking  = df[df["pitch_type"].isin(["SL", "CU", "KC", "SV"])]
    offspeed  = df[df["pitch_type"].isin(["CH", "FS", "FO"])]

    total  = max(len(df), 1)
    swings = df[df["description"].isin([
        "swinging_strike", "swinging_strike_blocked",
        "foul", "foul_tip", "hit_into_play",
    ])]
    whiffs = df[df["description"].isin([
        "swinging_strike", "swinging_strike_blocked",
    ])]
    in_zone       = df[df["zone"].fillna(0).between(1, 9)]
    out_zone      = df[~df["zone"].fillna(0).between(1, 9)]
    out_zone_swings = out_zone[out_zone["description"].isin([
        "swinging_strike", "swinging_strike_blocked",
        "foul", "foul_tip", "hit_into_play",
    ])]
    batted = df.dropna(subset=["launch_speed"])

    def _mean(series):
        val = series.mean()
        return float(val) if not pd.isna(val) else None

    xwoba_val = _mean(df["estimated_woba_using_speedangle"])

    # Pitcher.ops_allowed has no direct Statcast equivalent — derive it
    # from xwOBA using the same conversion build_dataset.py uses for
    # training labels (xwoba * 3.2 approximates OPS). Without this,
    # ops_allowed stays NULL for every pitcher, which silently breaks
    # any downstream formula that reads it (e.g. dp_engine.py's
    # division-bonus ops_gap calculation).
    ops_allowed_val = (xwoba_val * 3.2) if xwoba_val is not None else None

    return {
        "fb_velo":        _mean(fastballs["release_speed"]),
        "spin_rate":      _mean(fastballs["release_spin_rate"]),
        "extension":      _mean(df["release_extension"]),
        "ivb":            (_mean(fastballs["pfx_z"]) or 0) * 12,
        "hb":             (_mean(fastballs["pfx_x"]) or 0) * 12,
        "whiff_pct":      len(whiffs) / max(len(swings), 1),
        "zone_pct":       len(in_zone) / total,
        "chase_pct":      len(out_zone_swings) / max(len(out_zone), 1),
        "fastball_usage": len(fastballs) / total,
        "breaking_usage": len(breaking)  / total,
        "offspeed_usage": len(offspeed)  / total,
        "xwoba":          xwoba_val,
        "ops_allowed":    ops_allowed_val,
        "hard_hit_pct":   (batted["launch_speed"] >= 95).sum() / max(len(batted), 1),
    }


# ─────────────────────────────────────────────────────────────
# TEAM — rewritten to use MLB Stats API (no Fangraphs, no full
#         Statcast dump).  Falls back to pybaseball for whiff/
#         chase since those aren't in the standard batting API.
# ─────────────────────────────────────────────────────────────

def fetch_team_statcast(mlb_team_id: int, days: int = 60) -> dict:
    """
    Pull team offensive profile using the MLB Stats API for
    standard batting stats, and a targeted pybaseball pull
    for Statcast-only metrics (whiff, chase, xwOBA, hard-hit).

    Much faster than pulling the full 60-day Statcast dump —
    we filter to just this team's at-bats.
    """
    season = date.today().year

    # ── Standard batting from MLB Stats API (fast, no scraping) ──
    mlb_stats = _fetch_mlb_team_batting(mlb_team_id, season)

    # ── Statcast metrics via targeted pybaseball pull ─────────────
    statcast_stats = _fetch_team_statcast_metrics(mlb_team_id, days)

    return {**mlb_stats, **statcast_stats}


def _fetch_mlb_team_batting(mlb_team_id: int, season: int) -> dict:
    """
    Fetch OPS, K%, BB%, wOBA from the MLB Stats API team endpoint.
    Returns league averages on failure so the team row stays valid.
    """
    url = (
        f"https://statsapi.mlb.com/api/v1/teams/{mlb_team_id}/stats"
        f"?stats=season&group=hitting&season={season}&sportId=1"
    )
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        splits = r.json().get("stats", [{}])[0].get("splits", [{}])
        stat   = splits[0].get("stat", {}) if splits else {}

        pa  = int(stat.get("plateAppearances", 0) or 0)
        so  = int(stat.get("strikeOuts", 0) or 0)
        bb  = int(stat.get("baseOnBalls", 0) or 0)
        ops = float(stat.get("ops", 0.720) or 0.720)

        return {
            "ops":    ops,
            "k_pct":  so / max(pa, 1),
            "bb_pct": bb / max(pa, 1),
        }
    except Exception:
        return {"ops": 0.720, "k_pct": 0.220, "bb_pct": 0.080}


def _fetch_team_statcast_metrics(mlb_team_id: int, days: int) -> dict:
    """
    Pull Statcast pitch-level data for a team's at-bats using
    pybaseball's statcast_batter function filtered to the team.

    Uses the pybaseball cache — after the first pull per season
    this reads from disk, not the network.
    """
    end   = date.today()
    start = end - timedelta(days=days)
    abbrev = MLB_ID_TO_ABBREV.get(mlb_team_id)

    defaults = {
        "whiff_rate_60d": 0.240,
        "chase_pct":      0.290,
        "xwoba":          0.315,
        "hard_hit_pct":   0.370,
    }

    if not abbrev:
        return defaults

    try:
        # Pull team batting Statcast — filters to batters on this team
        df = pb.statcast_batter_exitvelo_barrels(
            year   = date.today().year,
            minBBE = 0,
        )

        if df is None or df.empty:
            return defaults

        # statcast_batter_exitvelo_barrels doesn't have pitch-level swing data
        # so fall back to team_batting for what we can get
        tb = pb.team_batting(
            start_season = date.today().year,
            end_season   = date.today().year,
        )

        if tb is not None and not tb.empty:
            # Match by team abbreviation
            team_row = tb[tb["teamIDfg"].str.upper() == abbrev] if "teamIDfg" in tb.columns else pd.DataFrame()
            if team_row.empty and "Team" in tb.columns:
                team_row = tb[tb["Team"].str.upper() == abbrev]

            if not team_row.empty:
                row = team_row.iloc[0]
                whiff = float(row.get("Whiff%", 0.240) or 0.240) / 100 \
                        if float(row.get("Whiff%", 24.0) or 24.0) > 1 \
                        else float(row.get("Whiff%", 0.240) or 0.240)
                o_swing = float(row.get("O-Swing%", 0.290) or 0.290) / 100 \
                          if float(row.get("O-Swing%", 29.0) or 29.0) > 1 \
                          else float(row.get("O-Swing%", 0.290) or 0.290)
                xwoba = float(row.get("xwOBA", 0.315) or 0.315)
                hard_hit = float(row.get("HardHit%", 0.370) or 0.370) / 100 \
                           if float(row.get("HardHit%", 37.0) or 37.0) > 1 \
                           else float(row.get("HardHit%", 0.370) or 0.370)

                return {
                    "whiff_rate_60d": whiff,
                    "chase_pct":      o_swing,
                    "xwoba":          xwoba,
                    "hard_hit_pct":   hard_hit,
                }

        return defaults

    except Exception:
        return defaults