# ml/features.py
import pandas as pd
import numpy as np
from database.models import Pitcher, Team

# Must exactly match FEATURE_COLS in ml/train.py and the columns
# produced by ml/build_dataset.py — any mismatch causes silent wrong predictions
FEATURE_COLS = [
    "p_whiff_pct",
    "p_zone_pct",
    "p_chase_pct",
    "p_fb_velo",
    "p_spin_rate",
    "p_extension",
    "p_xwoba_allowed",
    "p_hard_hit_pct",
    "p_fb_usage",
    "p_breaking_usage",
    "p_offspeed_usage",
    "opp_ops",
    "opp_whiff_rate",
    "opp_chase_rate",
    "opp_k_pct",
    "opp_hard_hit_pct",
    "opp_xwoba",
    "whiff_matchup",
    "chase_matchup",
    "ops_gap",
    "power_matchup",
]


def build_pitcher_features(pitcher_id: int, db_session) -> dict:
    """
    Pull Statcast-derived features for a single pitcher.
    Column names match what build_dataset.py produced for training.
    Field names verified against database/models.py Pitcher + StatcastMixin.
    """
    p = db_session.query(Pitcher).filter_by(id=pitcher_id).first()

    def _f(val, default):
        """Float with fallback — handles None and zero."""
        return float(val) if val is not None else default

    return {
        # StatcastMixin fields
        "p_whiff_pct":      _f(p.whiff_pct,       0.240) if p else 0.240,
        "p_zone_pct":       _f(p.zone_pct,         0.480) if p else 0.480,
        "p_chase_pct":      _f(p.chase_pct,        0.290) if p else 0.290,  # StatcastMixin.chase_pct
        "p_hard_hit_pct":   _f(p.hard_hit_pct,     0.370) if p else 0.370,
        "p_xwoba_allowed":  _f(p.xwoba,            0.315) if p else 0.315,  # StatcastMixin.xwoba
        # Pitcher-specific fields
        "p_fb_velo":        _f(p.fb_velo,           93.0) if p else 93.0,
        "p_spin_rate":      _f(p.spin_rate,         2250) if p else 2250,
        "p_extension":      _f(p.extension,          6.2) if p else 6.2,
        "p_fb_usage":       _f(p.fastball_usage,    0.35) if p else 0.35,
        "p_breaking_usage": _f(p.breaking_usage,    0.30) if p else 0.30,
        "p_offspeed_usage": _f(p.offspeed_usage,    0.15) if p else 0.15,
    }


def build_opponent_features(team_id: int, db_session) -> dict:
    """
    Pull batting profile for the opposing team.
    Column names match what build_dataset.py produced for training.
    Field names verified against database/models.py Team + StatcastMixin.
    """
    t = db_session.query(Team).filter_by(id=team_id).first()

    def _f(val, default):
        return float(val) if val is not None else default

    return {
        "opp_ops":          _f(t.ops,           0.720) if t else 0.720,
        "opp_whiff_rate":   _f(t.whiff_rate_60d, 0.240) if t else 0.240,  # Team.whiff_rate_60d
        "opp_chase_rate":   _f(t.chase_pct,      0.290) if t else 0.290,  # StatcastMixin.chase_pct
        "opp_k_pct":        _f(t.k_pct,          0.220) if t else 0.220,
        "opp_hard_hit_pct": _f(t.hard_hit_pct,   0.370) if t else 0.370,
        "opp_xwoba":        _f(t.xwoba,          0.315) if t else 0.315,  # StatcastMixin.xwoba
    }


def build_matchup_features(pitcher_id: int, team_id: int, db_session) -> pd.DataFrame:
    """
    Combine pitcher + opponent into a single feature vector
    ready to pass directly to the trained models.
    """
    p = build_pitcher_features(pitcher_id, db_session)
    o = build_opponent_features(team_id, db_session)

    row = {
        **p,
        **o,
        # Interaction terms — trained on these exact names
        "whiff_matchup":    p["p_whiff_pct"]    * o["opp_whiff_rate"],
        "chase_matchup":    p["p_chase_pct"]    * o["opp_chase_rate"],
        "ops_gap":          o["opp_ops"]        - (p["p_xwoba_allowed"] * 3.2),
        "power_matchup":    p["p_fb_velo"]      * o["opp_hard_hit_pct"],
    }

    # Return in exact column order the model expects
    return pd.DataFrame([row])[FEATURE_COLS]