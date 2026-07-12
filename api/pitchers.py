# api/pitchers.py
from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from database.models import Pitcher, GameStart, get_db_session
from optimizer.fatigue_model import build_fatigue_state, FatigueState
from optimizer.matchup_model import compute_matchup_score, PitcherProfile, OpponentProfile
from ml.predict import PitcherScoringPredictor


router = APIRouter(prefix="/pitchers", tags=["pitchers"])
predictor = PitcherScoringPredictor()   # loaded once at import time


# ─────────────────────────────────────────────────────────────
# RESPONSE SCHEMAS
# ─────────────────────────────────────────────────────────────

class FatigueResponse(BaseModel):
    pitcher_id:          int
    pitcher_name:        str
    as_of_date:          date
    days_since_last:     int
    pitch_count_7d:      int
    pitch_count_14d:     int
    pitch_count_28d:     int
    last_pitch_count:    int
    last_innings:        float
    last_velocity_delta: float
    rest_score:          float = Field(..., ge=0, le=1)
    workload_score:      float = Field(..., ge=0, le=1)
    recovery_score:      float = Field(..., ge=0, le=1)
    fatigue_bucket:      int   = Field(..., ge=0, le=4,
                                       description="Discretized fatigue level used by DP. 0=exhausted, 4=fresh")
    eligible_to_start:   bool


class PitcherSummaryResponse(BaseModel):
    id:                 int
    name:               str
    team_id:            Optional[int]
    throws:             str
    is_active:          bool
    fb_velo:            Optional[float]
    k_pct:              Optional[float]
    ops_allowed:        Optional[float]
    ops_allowed_60d:    Optional[float]
    xfip:               Optional[float]
    xfip_60d:           Optional[float]
    stats_as_of:        Optional[date]


class MatchupPreviewResponse(BaseModel):
    pitcher_id:         int
    pitcher_name:       str
    opponent_team_id:   int
    matchup_total:      float
    whiff_compat:       float
    ops_suppression:    float
    platoon:            float
    arsenal:            float
    ml_k_score:         float
    ml_ops_score:       float
    ml_wpa_estimate:    float
    ml_confidence:      float
    weights_used:       dict


class RotationResponse(BaseModel):
    team_id:    int
    as_of_date: date
    starters:   list[PitcherSummaryResponse]


# ─────────────────────────────────────────────────────────────
# DEPENDENCY: DB SESSION
# ─────────────────────────────────────────────────────────────

def get_db(session_factory=Depends(lambda: None)):
    """
    FastAPI dependency. session_factory injected at app startup
    via app.dependency_overrides in main.py.
    """
    with get_db_session(session_factory) as db:
        yield db


# ─────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────

@router.get("/{pitcher_id}", response_model=PitcherSummaryResponse)
def get_pitcher(pitcher_id: int, db=Depends(get_db)):
    """
    Returns static profile for a single pitcher.
    Stats are refreshed nightly by the data pipeline.
    """
    pitcher = db.query(Pitcher).filter_by(id=pitcher_id, is_active=True).first()
    if not pitcher:
        raise HTTPException(status_code=404, detail=f"Pitcher {pitcher_id} not found")

    return PitcherSummaryResponse(
        id              = pitcher.id,
        name            = pitcher.name,
        team_id         = pitcher.team_id,
        throws          = pitcher.throws,
        is_active       = pitcher.is_active,
        fb_velo         = pitcher.fb_velo,
        k_pct           = pitcher.k_pct,
        ops_allowed     = pitcher.ops_allowed,
        ops_allowed_60d = pitcher.ops_allowed_60d,
        xfip            = pitcher.xfip,
        xfip_60d        = pitcher.xfip_60d,
        stats_as_of     = pitcher.stats_as_of,
    )


@router.get("/{pitcher_id}/fatigue", response_model=FatigueResponse)
def get_pitcher_fatigue(
    pitcher_id: int,
    as_of:      Optional[date] = Query(default=None,
                                       description="Date to compute fatigue for. Defaults to today."),
    db=Depends(get_db),
):
    """
    Returns the full computed fatigue state for a pitcher.

    This is the same state the DP uses — calling this endpoint
    lets you inspect exactly what the optimizer sees before
    it makes an assignment decision.

    fatigue_bucket [0..4]:
      0 = exhausted (just pitched / short rest)
      1 = short rest (4 days — starts allowed, heavy penalty)
      2 = slightly tired (minor penalty)
      3 = normal rest (5 days — minimal penalty)
      4 = fully fresh (6+ days)
    """
    pitcher = db.query(Pitcher).filter_by(id=pitcher_id).first()
    if not pitcher:
        raise HTTPException(status_code=404, detail=f"Pitcher {pitcher_id} not found")

    target_date = as_of or date.today()

    state = build_fatigue_state(
        pitcher_id  = pitcher_id,
        as_of_date  = target_date,
        db_session  = db,
    )

    # Discretize to bucket (mirrors dp_engine.discretize_fatigue)
    from optimizer.dp_engine import discretize_fatigue, MIN_REST_DAYS
    bucket          = discretize_fatigue(state)
    eligible        = state.days_since_last >= MIN_REST_DAYS

    return FatigueResponse(
        pitcher_id          = pitcher_id,
        pitcher_name        = pitcher.name,
        as_of_date          = target_date,
        days_since_last     = state.days_since_last,
        pitch_count_7d      = state.pitch_count_7d,
        pitch_count_14d     = state.pitch_count_14d,
        pitch_count_28d     = state.pitch_count_28d,
        last_pitch_count    = state.last_pitch_count,
        last_innings        = state.last_innings,
        last_velocity_delta = state.last_velocity_delta,
        rest_score          = round(state.rest_score, 4),
        workload_score      = round(state.workload_score, 4),
        recovery_score      = round(state.recovery_score, 4),
        fatigue_bucket      = bucket,
        eligible_to_start   = eligible,
    )


@router.get("/team/{team_id}/rotation", response_model=RotationResponse)
def get_rotation(
    team_id:    int,
    active_only: bool = Query(default=True),
    db=Depends(get_db),
):
    """
    Returns all starting pitchers on a team's roster.
    Used by the optimizer at startup to build its pitcher list.
    """
    query = (
        db.query(Pitcher)
        .filter_by(team_id=team_id, is_starter=True)
    )
    if active_only:
        query = query.filter_by(is_active=True)

    pitchers = query.order_by(Pitcher.xfip).all()

    if not pitchers:
        raise HTTPException(
            status_code=404,
            detail=f"No active starters found for team {team_id}"
        )

    return RotationResponse(
        team_id    = team_id,
        as_of_date = date.today(),
        starters   = [
            PitcherSummaryResponse(
                id              = p.id,
                name            = p.name,
                team_id         = p.team_id,
                throws          = p.throws,
                is_active       = p.is_active,
                fb_velo         = p.fb_velo,
                k_pct           = p.k_pct,
                ops_allowed     = p.ops_allowed,
                ops_allowed_60d = p.ops_allowed_60d,
                xfip            = p.xfip,
                xfip_60d        = p.xfip_60d,
                stats_as_of     = p.stats_as_of,
            )
            for p in pitchers
        ],
    )


@router.get("/{pitcher_id}/matchup/{opponent_team_id}", response_model=MatchupPreviewResponse)
def get_matchup_preview(
    pitcher_id:       int,
    opponent_team_id: int,
    db=Depends(get_db),
):
    """
    Returns the full matchup score breakdown for a specific
    pitcher vs. opponent pairing.

    This is what the optimizer scores internally — exposing it
    here lets coaches inspect and challenge any assignment.

    ML coefficients are fetched live from the active model version.
    """
    from database.models import Team

    pitcher  = db.query(Pitcher).filter_by(id=pitcher_id, is_active=True).first()
    opponent = db.query(Team).filter_by(id=opponent_team_id).first()

    if not pitcher:
        raise HTTPException(status_code=404, detail=f"Pitcher {pitcher_id} not found")
    if not opponent:
        raise HTTPException(status_code=404, detail=f"Team {opponent_team_id} not found")

    # Build profile objects (mirrors what dp_engine.py does)
    pitcher_profile = _pitcher_to_profile(pitcher)
    opponent_profile = _team_to_profile(opponent)

    # ML coefficients
    coeffs = predictor.predict(pitcher_id, opponent_team_id, db)
    ml_weights = {
        "whiff_weight":   coeffs.k_score,
        "ops_weight":     coeffs.ops_score,
        "platoon_weight": 0.15,
        "arsenal_weight": 1.0 - coeffs.confidence,
    }

    breakdown = compute_matchup_score(
        pitcher         = pitcher_profile,
        opponent        = opponent_profile,
        ml_coefficients = ml_weights,
    )

    return MatchupPreviewResponse(
        pitcher_id       = pitcher_id,
        pitcher_name     = pitcher.name,
        opponent_team_id = opponent_team_id,
        matchup_total    = breakdown["total"],
        whiff_compat     = breakdown["whiff_compat"],
        ops_suppression  = breakdown["ops_suppression"],
        platoon          = breakdown["platoon"],
        arsenal          = breakdown["arsenal"],
        ml_k_score       = round(coeffs.k_score, 4),
        ml_ops_score     = round(coeffs.ops_score, 4),
        ml_wpa_estimate  = round(coeffs.wpa_estimate, 4),
        ml_confidence    = round(coeffs.confidence, 4),
        weights_used     = breakdown["weights_used"],
    )


@router.get("/{pitcher_id}/starts", response_model=list)
def get_recent_starts(
    pitcher_id: int,
    limit:      int  = Query(default=10, ge=1, le=50),
    db=Depends(get_db),
):
    """
    Returns recent GameStart rows for a pitcher.
    Useful for validating fatigue state and viewing pitch count history.
    """
    pitcher = db.query(Pitcher).filter_by(id=pitcher_id).first()
    if not pitcher:
        raise HTTPException(status_code=404, detail=f"Pitcher {pitcher_id} not found")

    starts = (
        db.query(GameStart)
        .filter_by(pitcher_id=pitcher_id)
        .order_by(GameStart.game_date.desc())
        .limit(limit)
        .all()
    )

    return [
        {
            "game_date":          s.game_date,
            "opponent_team_id":   s.opponent_team_id,
            "pitch_count":        s.pitch_count,
            "innings_pitched":    s.innings_pitched,
            "game_score":         s.game_score,
            "velocity_delta":     s.velocity_delta,
            "k_pct_actual":       s.k_pct_actual,
            "ops_allowed_actual": s.ops_allowed_actual,
            "wpa":                s.wpa,
            "was_recommended":    s.was_recommended,
        }
        for s in starts
    ]


# ─────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────

def _pitcher_to_profile(p: Pitcher, db_session=None, as_of_date=None) -> PitcherProfile:
    """Convert ORM Pitcher → matchup_model PitcherProfile."""
    from optimizer.matchup_model import PitcherProfile
    from optimizer.fatigue_model import build_fatigue_state

    return PitcherProfile(
        team_id               = p.team_id,
        id                    = p.id,
        current_fatigue_state = build_fatigue_state(
            pitcher_id = p.id,
            as_of_date = as_of_date,
            db_session = db_session,
        ) if db_session and as_of_date else None,
        k_pct = p.k_pct if p.k_pct is not None else 0.22,
        ops_allowed         = p.ops_allowed         or 0.720,
        whiff_pct           = p.whiff_pct           or 0.115,
        zone_pct            = p.zone_pct            or 0.47,
        chase_pct           = p.chase_pct           or 0.31,
        fb_velo             = p.fb_velo             or 92.0,
        extension           = p.extension           or 6.5,
        ops_allowed_vs_rhb  = p.ops_allowed_vs_rhb  or 0.720,
        ops_allowed_vs_lhb  = p.ops_allowed_vs_lhb  or 0.720,
        fastball_usage      = p.fastball_usage       or 0.55,
        breaking_usage      = p.breaking_usage       or 0.30,
        offspeed_usage      = p.offspeed_usage       or 0.15,
    )


def _team_to_profile(t) -> OpponentProfile:
    """Convert ORM Team → matchup_model OpponentProfile."""
    from optimizer.matchup_model import OpponentProfile
    return OpponentProfile(
        team_id             = t.id,
        ops                 = t.ops                 or 0.720,
        whiff_rate          = t.whiff_rate_60d      or 0.240,
        chase_rate          = t.chase_pct           or 0.29,
        k_pct               = t.k_pct               or 0.22,
        woba                = t.woba                or 0.315,
        xwoba               = t.xwoba               or 0.315,
        hard_hit_pct        = t.hard_hit_pct        or 0.37,
        rhb_pct             = t.rhb_pct             or 0.60,
        lhb_pct             = t.lhb_pct             or 0.40,
        ops_vs_fastball     = t.ops_vs_fastball     or 0.760,
        ops_vs_breaking     = t.ops_vs_breaking     or 0.670,
        ops_vs_offspeed     = t.ops_vs_offspeed     or 0.700,
    )