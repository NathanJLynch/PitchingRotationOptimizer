# optimizer/fatigue_model.py
import numpy as np
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
import logging



logger = logging.getLogger(__name__)
# ─────────────────────────────────────────────────────────────
# DATA CONTRACTS
# ─────────────────────────────────────────────────────────────

@dataclass
class PitchingOuting:
    """Single start record — stored in DB, loaded per pitcher."""
    game_date:      date
    pitch_count:    int
    innings_pitched: float
    game_score:     int     # Bill James game score — proxy for exertion quality
    velocity_delta: float   # avg velo vs. season avg (negative = fatigued)


@dataclass
class FatigueState:
    """
    Full fatigue state for a pitcher at a point in time.
    Computed fresh each optimizer run from recent outings.
    Passed into discretize_fatigue() in dp_engine.py.
    """
    pitcher_id:         int
    as_of_date:         date

    days_since_last:    int         # days since most recent start
    pitch_count_7d:     int         # total pitches thrown in last 7 days
    pitch_count_14d:    int         # total pitches thrown in last 14 days
    pitch_count_28d:    int         # total pitches thrown in last 28 days

    last_pitch_count:   int         # pitch count in most recent outing
    last_innings:       float       # IP in most recent outing
    last_velocity_delta: float      # velo drop in most recent outing

    rest_score:         float       # [0, 1] — 1.0 = fully rested, 0.0 = exhausted
    workload_score:     float       # [0, 1] — 1.0 = light recent load
    recovery_score:     float       # [0, 1] — composite of rest + workload

    recent_outings:     list = field(default_factory=list)  # list[PitchingOuting]


# ─────────────────────────────────────────────────────────────
# CONSTANTS
#
# Calibrated to MLB pitcher population.
# Sources: FanGraphs pitch count distributions, BP PECOTA workload data.
# ─────────────────────────────────────────────────────────────

# Rest thresholds (days since last start)
REST_EXHAUSTED     = 3    # < 4 days: cannot start, hard constraint
REST_SHORT         = 4    # 4 days: starts allowed, significant penalty
REST_NORMAL        = 5    # 5 days: standard modern rotation
REST_EXTRA         = 6    # 6 days: slight freshness bonus
REST_FULL          = 8    # 8+ days: fully recovered (skip start / IL return)

# Pitch count thresholds
PC_LIGHT_OUTING    = 75   # bullpen day / opener
PC_NORMAL_OUTING   = 95   # quality start range
PC_HIGH_OUTING     = 110  # full workload
PC_EXTREME_OUTING  = 120  # high-stress, accelerates fatigue

# Rolling workload limits (MLB pitcher health guidelines)
PC_LIMIT_7D        = 115  # soft limit: > 115 in 7 days → penalty
PC_LIMIT_14D       = 200  # soft limit: > 200 in 14 days → penalty
PC_LIMIT_28D       = 380  # soft limit: > 380 in 28 days → penalty

# Decay weights for rolling pitch count (more recent = more weight)
# Applied to outings sorted newest-first
DECAY_WEIGHTS_7D   = [1.00, 0.50]             # last 2 outings within 7 days
DECAY_WEIGHTS_14D  = [1.00, 0.75, 0.40]       # last 3 outings within 14 days
DECAY_WEIGHTS_28D  = [1.00, 0.80, 0.55, 0.30] # last 4 outings within 28 days

# Penalty magnitudes — how much fatigue reduces the DP score
PENALTY_BASE       = 0.05   # minimum penalty even for fully rested pitcher
PENALTY_SHORT_REST = 0.35   # starting on 4 days rest
PENALTY_OVERWORKED = 0.25   # rolling workload over limit
PENALTY_VELO_DROP  = 0.15   # pitcher showing velocity decline


# ─────────────────────────────────────────────────────────────
# REST SCORE
#
# Captures the recovery from the most recent outing.
# Non-linear: going from 4 → 5 days rest is a bigger
# improvement than going from 7 → 8 days.
# ─────────────────────────────────────────────────────────────

def compute_rest_score(days_since_last: int, last_pitch_count: int) -> float:
    """
    Returns [0, 1]. 1.0 = fully rested, 0.0 = cannot pitch.

    Accounts for the fact that a 115-pitch outing on 5 days rest
    still leaves a pitcher more fatigued than a 75-pitch outing
    on 5 days rest.
    """
    if days_since_last < REST_EXHAUSTED:
        return 0.0   # hard constraint enforced upstream in dp_engine.py

    # Base rest curve: asymptotic recovery toward 1.0
    # Inflection at REST_NORMAL (5 days), plateau at REST_FULL (8 days)
    base_rest = 1.0 - np.exp(-0.55 * (days_since_last - REST_EXHAUSTED))
    base_rest = float(np.clip(base_rest, 0.0, 1.0))

    # Pitch count modifier: high-pitch outings slow recovery
    # 95 pitches = neutral modifier (1.0)
    # 115 pitches = 15% slower recovery
    # 75 pitches  = 10% faster recovery
    pc_modifier = 1.0 - 0.0015 * max(last_pitch_count - PC_NORMAL_OUTING, 0)
    pc_modifier = float(np.clip(pc_modifier, 0.70, 1.10))

    rest_score = base_rest * pc_modifier
    return float(np.clip(rest_score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────
# WORKLOAD SCORE
#
# Captures cumulative stress across the 7/14/28-day windows.
# A pitcher can have great rest (5 days since last start) but
# still be worn down from a heavy recent schedule.
# ─────────────────────────────────────────────────────────────

def compute_workload_score(
    pitch_count_7d:  int,
    pitch_count_14d: int,
    pitch_count_28d: int,
) -> float:
    """
    Returns [0, 1]. 1.0 = light recent load, 0.0 = heavily overworked.
    """

    # Each window contributes a usage ratio: actual / limit
    # Over 1.0 = overworked, under 1.0 = within normal range
    ratio_7d  = pitch_count_7d  / PC_LIMIT_7D
    ratio_14d = pitch_count_14d / PC_LIMIT_14D
    ratio_28d = pitch_count_28d / PC_LIMIT_28D

    # Weight recent windows more heavily
    # 7-day is most predictive of next-start performance
    weighted_ratio = (
        0.55 * ratio_7d
        + 0.30 * ratio_14d
        + 0.15 * ratio_28d
    )

    # Convert to score: 1.0 at or below limits, decays as load increases
    workload_score = 1.0 - np.clip(weighted_ratio - 0.60, 0, 0.40) / 0.40
    # Explanation of clipping:
    #   - Below 60% of limits → no penalty, full score
    #   - 60–100% of limits   → linear decay from 1.0 → 0.0
    #   - Above 100%          → floored at 0.0

    return float(np.clip(workload_score, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────
# VELOCITY DELTA MODIFIER
#
# Velocity decline mid-outing or start-over-start is an
# early indicator of fatigue that precedes stat degradation.
# Only available if Statcast pitch-level data is loaded.
# ─────────────────────────────────────────────────────────────

def compute_velocity_modifier(last_velocity_delta: float) -> float:
    """
    last_velocity_delta: difference between pitcher's season-avg
    fastball velo and most recent start avg fastball velo.
    Negative = throwing softer than normal (fatigued signal).

    Returns a multiplier [0.80, 1.05]:
      -3.0 mph delta → 0.80 (strong fatigue signal)
       0.0 mph delta → 1.00 (baseline)
      +1.0 mph delta → 1.05 (fresher than normal)
    """
    # Linear interpolation with clipping
    if last_velocity_delta >= 0:
        modifier = 1.0 + 0.05 * min(last_velocity_delta, 1.0)
    else:
        modifier = 1.0 + 0.067 * last_velocity_delta   # negative slope
    return float(np.clip(modifier, 0.75, 1.05))


# ─────────────────────────────────────────────────────────────
# RECOVERY SCORE
#
# Single composite [0, 1] value passed to discretize_fatigue()
# in dp_engine.py. Combines rest + workload + velocity signal.
# ─────────────────────────────────────────────────────────────

def compute_recovery_score(
    rest_score:          float,
    workload_score:      float,
    velocity_modifier:   float,
) -> float:
    """
    Combines the three sub-scores into a single recovery value.

    rest_score drives the majority: short rest is the biggest
    single predictor of performance degradation.
    workload penalizes accumulated wear across the schedule.
    velocity is a signal amplifier, not a primary driver.
    """
    base = (
        0.60 * rest_score
        + 0.40 * workload_score
    )
    # Velocity modifier scales the combined score, not added to it
    # This preserves the [0,1] range more reliably
    recovery = base * velocity_modifier
    return float(np.clip(recovery, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────
# FATIGUE PENALTY
#
# This is what dp_engine.py calls inside the DP loop.
# Converts a fatigue level bucket [0..4] into a score penalty.
#
# The penalty is subtracted from the total game score:
#   dp[i][j][f] = matchup_score + division_bonus - fatigue_penalty
# ─────────────────────────────────────────────────────────────

def fatigue_penalty(pitcher, fatigue_level: int) -> float:
    """
    pitcher:       PitcherProfile (from matchup_model.py)
    fatigue_level: int [0..4], discretized bucket from dp_engine.py
                   0 = exhausted, 4 = fully fresh

    Returns a non-negative penalty value subtracted from DP score.

    Penalty curve:
      fatigue 0 → 0.40  (exhausted: hard to justify starting)
      fatigue 1 → 0.25  (short rest: meaningful degradation)
      fatigue 2 → 0.12  (slightly tired: minor penalty)
      fatigue 3 → 0.04  (normal rest: minimal penalty)
      fatigue 4 → 0.00  (fully rested: no penalty)
    """

    # Base penalty by fatigue bucket
    BASE_PENALTIES = {
        0: 0.40,
        1: 0.25,
        2: 0.12,
        3: 0.04,
        4: 0.00,
    }
    base = BASE_PENALTIES.get(fatigue_level, 0.0)

    # Pitcher-specific modifier: high-inning workhorses
    # recover faster than finesse pitchers (lower extension,
    # slower velocity → more mechanical stress per pitch)
    durability_modifier = _durability_modifier(pitcher)

    penalty = base * durability_modifier
    return float(np.clip(penalty + PENALTY_BASE, PENALTY_BASE, 0.50))


def _durability_modifier(pitcher) -> float:
    """
    Scale penalty up/down based on pitcher's physical profile.
    High-velocity, high-extension pitchers have more physical
    stress per pitch → recover slower → modifier > 1.0.
    Soft-tossers with low extension → modifier < 1.0.
    """
    # Velo: > 95 mph adds mechanical stress
    velo_factor = 1.0 + 0.02 * max(pitcher.fb_velo - 93.0, 0)

    # Extension: longer extension = more arm stress per pitch
    # League avg ~6.5ft; elite ~7.2ft
    ext_factor = 1.0 + 0.05 * max(pitcher.extension - 6.5, 0)

    modifier = velo_factor * ext_factor
    return float(np.clip(modifier, 0.85, 1.30))


# ─────────────────────────────────────────────────────────────
# FATIGUE STATE BUILDER
#
# Called once per pitcher at the start of each optimizer run.
# Loads recent outings from the DB and computes the full state.
# ─────────────────────────────────────────────────────────────

def build_fatigue_state(
    pitcher_id:   int,
    as_of_date:   date,
    db_session,
) -> FatigueState:
    
    """
    Pulls recent outings for pitcher_id and computes all
    fatigue sub-scores. Returns a FatigueState ready for
    discretize_fatigue() in dp_engine.py.
    """
    from database.models import GameStart   # local import avoids circular deps

    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.date()  # convert datetime → date

    # Load last 6 starts (enough to cover 28-day window for any rotation slot)
    recent_rows = (
        db_session.query(GameStart)
        .filter(GameStart.pitcher_id == pitcher_id)
        .filter(GameStart.game_date < as_of_date)
        .order_by(GameStart.game_date.desc())
        .limit(6)
        .all()
    )


    if not recent_rows:
        # No recent history: treat as fully rested (spring training / IL return)
        return FatigueState(
            pitcher_id          = pitcher_id,
            as_of_date          = as_of_date,
            days_since_last     = 999,
            pitch_count_7d      = 0,
            pitch_count_14d     = 0,
            pitch_count_28d     = 0,
            last_pitch_count    = 0,
            last_innings        = 0.0,
            last_velocity_delta = 0.0,
            rest_score          = 1.0,
            workload_score      = 1.0,
            recovery_score      = 1.0,
            recent_outings      = [],
        )

    outings = [
        PitchingOuting(
            game_date       = row.game_date,
            pitch_count     = row.pitch_count,
            innings_pitched = row.innings_pitched,
            game_score      = row.game_score,
            velocity_delta  = row.velocity_delta or 0.0,
        )
        for row in recent_rows
    ]

    most_recent = outings[0]
    days_since  = (as_of_date - most_recent.game_date).days

    # Rolling pitch counts by window
    def pitches_in_window(days: int) -> int:
        cutoff = as_of_date - timedelta(days=days)
        return sum(o.pitch_count for o in outings if o.game_date >= cutoff)

    pc_7d  = pitches_in_window(7)
    pc_14d = pitches_in_window(14)
    pc_28d = pitches_in_window(28)

    # Sub-scores
    rest_score      = compute_rest_score(days_since, most_recent.pitch_count)
    workload_score  = compute_workload_score(pc_7d, pc_14d, pc_28d)
    velo_modifier   = compute_velocity_modifier(most_recent.velocity_delta)
    recovery_score  = compute_recovery_score(rest_score, workload_score, velo_modifier)

    return FatigueState(
        pitcher_id          = pitcher_id,
        as_of_date          = as_of_date,
        days_since_last     = days_since,
        pitch_count_7d      = pc_7d,
        pitch_count_14d     = pc_14d,
        pitch_count_28d     = pc_28d,
        last_pitch_count    = most_recent.pitch_count,
        last_innings        = most_recent.innings_pitched,
        last_velocity_delta = most_recent.velocity_delta,
        rest_score          = rest_score,
        workload_score      = workload_score,
        recovery_score      = recovery_score,
        recent_outings      = outings,
    )


# ─────────────────────────────────────────────────────────────
# FATIGUE STATE PROJECTOR
#
# The DP needs to simulate fatigue FORWARD through the schedule.
# Given a pitcher's state today, what will it look like after
# their next start on date X with Y pitches thrown?
# ─────────────────────────────────────────────────────────────

def project_fatigue_state(
    current_state:    FatigueState,
    start_date:       date,
    projected_pitches: int = 95,
) -> FatigueState:
    """
    Simulates a future outing and returns the resulting FatigueState.
    Used by dp_engine.py's valid_transitions() to project
    what fatigue level a pitcher will be at in future games.

    projected_pitches defaults to 95 (league-average QS workload).
    The ML model's innings prediction can override this at runtime.
    """

    # Add projected outing to recent outings list
    projected_outing = PitchingOuting(
        game_date       = start_date,
        pitch_count     = projected_pitches,
        innings_pitched = projected_pitches / 15.5,   # rough IP estimate
        game_score      = 50,                          # neutral game score
        velocity_delta  = 0.0,                         # assume no velocity change
    )

    new_outings = [projected_outing] + current_state.recent_outings[:5]

    # Recalculate from projected state
    as_of_next = start_date + timedelta(days=1)
    days_since  = 1   # just pitched

    def pitches_in_window(days: int) -> int:
        cutoff = as_of_next - timedelta(days=days)
        return sum(o.pitch_count for o in new_outings if o.game_date >= cutoff)

    pc_7d  = pitches_in_window(7)
    pc_14d = pitches_in_window(14)
    pc_28d = pitches_in_window(28)

    rest_score     = compute_rest_score(days_since, projected_pitches)
    workload_score = compute_workload_score(pc_7d, pc_14d, pc_28d)
    velo_modifier  = compute_velocity_modifier(0.0)
    recovery_score = compute_recovery_score(rest_score, workload_score, velo_modifier)

    return FatigueState(
        pitcher_id          = current_state.pitcher_id,
        as_of_date          = as_of_next,
        days_since_last     = days_since,
        pitch_count_7d      = pc_7d,
        pitch_count_14d     = pc_14d,
        pitch_count_28d     = pc_28d,
        last_pitch_count    = projected_pitches,
        last_innings        = projected_pitches / 15.5,
        last_velocity_delta = 0.0,
        rest_score          = rest_score,
        workload_score      = workload_score,
        recovery_score      = recovery_score,
        recent_outings      = new_outings,
    )


# ─────────────────────────────────────────────────────────────
# FATIGUE STATE UPDATER
#
# Called AFTER a game completes to write updated fatigue
# back to the database. Separate from the projector —
# this uses actual pitch count, not projected.
# ─────────────────────────────────────────────────────────────

def update_fatigue(
    pitcher_id:     int,
    game_date:      date,
    actual_pitches: int,
    actual_innings: float,
    velocity_delta: float,
    db_session,
) -> FatigueState:
    """
    Post-game update. Writes the actual outing to GameStart
    then recomputes and returns the updated FatigueState.
    Called by api/games.py after each game result is logged.
    """
    from database.models import GameStart

    # Upsert the actual outing
    existing = (
        db_session.query(GameStart)
        .filter_by(pitcher_id=pitcher_id, game_date=game_date)
        .first()
    )
    if existing:
        existing.pitch_count     = actual_pitches
        existing.innings_pitched = actual_innings
        existing.velocity_delta  = velocity_delta
    else:
        db_session.add(GameStart(
            pitcher_id      = pitcher_id,
            game_date       = game_date,
            pitch_count     = actual_pitches,
            innings_pitched = actual_innings,
            velocity_delta  = velocity_delta,
            game_score      = _estimate_game_score(actual_pitches, actual_innings),
        ))
    db_session.commit()

    # Recompute from fresh DB state
    return build_fatigue_state(
        pitcher_id  = pitcher_id,
        as_of_date  = game_date + timedelta(days=1),
        db_session  = db_session,
    )


def _estimate_game_score(pitch_count: int, innings_pitched: float) -> int:
    """
    Rough Bill James game score estimate when actual isn't available.
    Real game score also factors in K, BB, H — this is a workload proxy only.
    """
    return int(40 + innings_pitched * 3 + (pitch_count / 10))