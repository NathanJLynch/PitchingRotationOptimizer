# optimizer/dp_engine.py
import numpy as np
from dataclasses import dataclass, field
from datetime import date
from typing import Optional
from itertools import product

from optimizer.matchup_model  import compute_matchup_score, PitcherProfile, OpponentProfile
from optimizer.division_bonus import DivisionBonusCalculator, DivisionBonusResult
from optimizer.fatigue_model  import fatigue_penalty, FatigueState
from ml.predict               import PitcherScoringPredictor


# ─────────────────────────────────────────────────────────────
# DATA CONTRACTS
# ─────────────────────────────────────────────────────────────

@dataclass
class Game:
    game_id:          int
    date:             date
    opponent_team_id: int
    opponent:         OpponentProfile
    is_home:          bool
    series_game_num:  int
    h2h_remaining:    int

@dataclass
class AssignedStart:
    game:             Game
    pitcher:          PitcherProfile
    fatigue_level:    int
    matchup_score:    float
    division_bonus:   float
    fatigue_penalty:  float
    total_score:      float
    score_breakdown:  dict


@dataclass
class OptimizationResult:
    assignments:      list
    total_score:      float
    run_date:         date
    horizon_days:     int
    our_team_id:      int


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

FATIGUE_LEVELS   = 5    # 0 = just pitched → 4 = fully fresh
MIN_REST_DAYS    = 4    # hard constraint: cannot start at fatigue < this many days rest
NEG_INF          = -1e9


# ─────────────────────────────────────────────────────────────
# FATIGUE DISCRETIZER
# ─────────────────────────────────────────────────────────────

def discretize_fatigue(fatigue_state: FatigueState) -> int:
    """
    Maps a FatigueState (or raw days-since-last-start int) to bucket 0–4.
    0 = just pitched / ineligible, 4 = 6+ days rest / fully fresh.
    """
    if isinstance(fatigue_state, int):
        days = fatigue_state
    else:
        days = fatigue_state.days_since_last

    # Explicit mapping so the buckets match the comment and fatigue_penalty()
    #   0–1 days  → 0  (just pitched, ineligible)
    #   2 days    → 1
    #   3 days    → 2
    #   4 days    → 3  (short rest, meaningful penalty)
    #   5 days    → 3  (normal rest, minor penalty)
    #   6+ days   → 4  (fully fresh, no penalty)
    if days <= 1:
        return 0
    elif days == 2:
        return 1
    elif days == 3:
        return 2
    elif days <= 5:
        return 3
    else:
        return 4


def fatigue_bucket_to_days(bucket: int) -> int:
    """Inverse mapping for projecting forward."""
    return bucket  # bucket IS days-since-last, capped at FATIGUE_LEVELS-1


# ─────────────────────────────────────────────────────────────
# DP SOLVER
#
# STATE REDESIGN:
# The previous version tracked dp[i][j][f] — only ONE pitcher's
# fatigue. This is insufficient: the DP needs to know every
# pitcher's fatigue to know who's eligible on day i.
#
# New approach: dp[i][fatigue_vector] where fatigue_vector is a
# tuple of length n_pitchers, each entry in [0..FATIGUE_LEVELS-1].
#
# This is exponential in n_pitchers (5^n_pitchers states per game),
# which is fine for realistic rotation sizes (5-7 starters).
# For n_pitchers > 8, this would need a different approach
# (e.g. only track the "recently used" subset).
#
# Transition i → i+1:
#   - Choose which pitcher j starts game i (must have fatigue >= MIN_REST_DAYS bucket)
#   - j's fatigue resets to 0 for the next state
#   - everyone else's fatigue increments by days_gap (capped at 4)
# ─────────────────────────────────────────────────────────────

# Minimum fatigue bucket required to start (4 days rest = bucket 3)
ELIGIBLE_BUCKET = min(MIN_REST_DAYS, FATIGUE_LEVELS - 1)


def _advance_fatigue(fatigue_vector: tuple, started_idx: Optional[int], days_gap: int) -> tuple:
    """
    Given the fatigue vector BEFORE game i, and which pitcher (if any)
    started game i, compute the fatigue vector for game i+1.

    started_idx pitcher → resets to 0, then advances by (days_gap - 1)
                          since rest begins the day after pitching
    everyone else        → advances by days_gap, capped at FATIGUE_LEVELS-1
    """
    new_vec = list(fatigue_vector)
    for j in range(len(new_vec)):
        if j == started_idx:
            # Pitched today: 0 days rest now, gains (days_gap - 1) by next game
            new_vec[j] = min(max(days_gap - 1, 0), FATIGUE_LEVELS - 1)
        else:
            new_vec[j] = min(new_vec[j] + days_gap, FATIGUE_LEVELS - 1)
    return tuple(new_vec)


def _eligible_pitchers(fatigue_vector, pitchers=None, game=None, bonus_result=None):
    eligible = [j for j, f in enumerate(fatigue_vector) if f >= ELIGIBLE_BUCKET]

    if not eligible:
        return [int(np.argmax(fatigue_vector))]

    if pitchers is not None and game is not None and bonus_result is not None:
        if bonus_result.regime == "pennant_race":

            # How strict the quality filter is:
            # Lower = only good pitchers start big games
            # Higher = more pitchers allowed through
            buffer = 0.08

            quality_eligible = [
                j for j in eligible
                if (pitchers[j].ops_allowed or 0.80) <= game.opponent.ops + buffer
            ]
            if quality_eligible:
                return quality_eligible

    return eligible


# ─────────────────────────────────────────────────────────────
# SCORING FUNCTION
# ─────────────────────────────────────────────────────────────

def compute_score(
    pitcher:         PitcherProfile,
    game:            Game,
    fatigue_level:   int,
    bonus_result:    DivisionBonusResult,
    ml_coefficients: Optional[dict] = None,
) -> tuple[float, dict]:
    """
    Returns (score, breakdown_dict).
    Score = matchup_score + division_bonus - fatigue_penalty
    """
    matchup = compute_matchup_score(
        pitcher          = pitcher,
        opponent         = game.opponent,
        ml_coefficients  = ml_coefficients,
    )

    division_bonus = 0.0
    if bonus_result.regime == "pennant_race":
        ops_gap = max(game.opponent.ops - pitcher.ops_allowed, 0.0)
        division_bonus = bonus_result.bonus_multiplier * ops_gap * matchup["ops_suppression"]
    elif bonus_result.regime == "clinched":
        division_bonus = 0.05 * matchup["total"]

    fat_penalty = fatigue_penalty(pitcher, fatigue_level)
    total = matchup["total"] + division_bonus - fat_penalty

    breakdown = {
        "matchup_total":      round(matchup["total"], 4),
        "matchup_detail":     matchup,
        "division_bonus":     round(division_bonus, 4),
        "division_regime":    bonus_result.regime,
        "division_urgency":   round(bonus_result.urgency_score, 4),
        "fatigue_level":      fatigue_level,
        "fatigue_penalty":    round(fat_penalty, 4),
        "total":              round(total, 4),
    }
    return total, breakdown


# ─────────────────────────────────────────────────────────────
# MAIN SOLVER
# ─────────────────────────────────────────────────────────────

def solve(
    games:        list,
    pitchers:     list,
    our_team_id:  int,
    db_session,
    run_date:     Optional[date] = None,
) -> OptimizationResult:
    
        # In dp_engine.py, inside solve(), before the main DP loop
    wrong_team = [p for p in pitchers if p.team_id != our_team_id]
    if wrong_team:
        raise ValueError(
            f"Pitchers {[p.id for p in wrong_team]} do not belong to team {our_team_id}. "
            f"solve() must be called with pitchers scoped to our_team_id."
        )
    def _rank_game_leverage(games, bonus_results):
        """
        Returns a list of (game_index, leverage_score) sorted high to low.
        Used to identify which games most need quality starters.
        """
        leverage = []
        for i, (game, bonus) in enumerate(zip(games, bonus_results)):
            # High OPS opponent + pennant race = high leverage
            opp_ops    = game.opponent.ops
            race_mult  = 1.5 if bonus.regime == "pennant_race" else 1.0
            leverage.append((i, opp_ops * race_mult))
        return sorted(leverage, key=lambda x: x[1], reverse=True)

    run_date   = run_date or date.today()
    n_games    = len(games)
    n_pitchers = len(pitchers)

    if n_games == 0 or n_pitchers == 0:
        raise ValueError("Need at least one game and one pitcher to optimize.")

    # ── Initial fatigue vector from each pitcher's current state ──
    initial_fatigue = tuple(
        discretize_fatigue(p.current_fatigue_state) if p.current_fatigue_state is not None
        else ELIGIBLE_BUCKET   # treat unknown fatigue as fully rested
        for p in pitchers
    )

    # ── Precompute days-gap between consecutive games ─────────────
    days_gaps = []
    for i in range(n_games):
        if i + 1 < n_games:
            gap = (games[i + 1].date - games[i].date).days
        else:
            gap = 1
        days_gaps.append(max(gap, 1))

    # ── Precompute per-game, per-pitcher scores (independent of fatigue vector,
    #    EXCEPT the fatigue_level term which depends on the pitcher's OWN
    #    fatigue at that point — but since transitions reset/advance
    #    deterministically, we compute score for every possible fatigue bucket
    #    a pitcher could be in, then look up by bucket at solve time) ──
    predictor  = PitcherScoringPredictor()
    bonus_calc = DivisionBonusCalculator(our_team_id=our_team_id, db_session=db_session)

    bonus_results = []
    for game in games:
        bonus_results.append(bonus_calc.compute(
            opponent_team_id = game.opponent_team_id,
            as_of_date       = game.date,
            h2h_remaining    = game.h2h_remaining,
        ))
    # score_cache[i][j][f] = (score, breakdown) for pitcher j starting game i at fatigue f
    score_cache = {}
    for i, game in enumerate(games):
        for j, pitcher in enumerate(pitchers):
            ml_coeffs = predictor.predict(
                pitcher_id = pitcher.id,
                team_id    = game.opponent_team_id,
                db_session = db_session,
            )
            ml_weight_dict = {
                "whiff_weight":   ml_coeffs.k_score,
                "ops_weight":     ml_coeffs.ops_score,
                "platoon_weight": 0.15,
                "arsenal_weight": 1.0 - ml_coeffs.confidence,
            } if ml_coeffs is not None else None
            for f in range(FATIGUE_LEVELS):
                score, breakdown = compute_score(
                    pitcher         = pitcher,
                    game            = game,
                    fatigue_level   = f,
                    bonus_result = bonus_results[i],
                    ml_coefficients = ml_weight_dict,
                )
                score_cache[(i, j, f)] = (score, breakdown)

    # ── DP over fatigue vectors ────────────────────────────────────
    # dp[i][fatigue_vector] = (best_value, best_pitcher_idx)
    # Memoized recursively (top-down) since the state space can be
    # large but only reachable states get visited.

    memo = {}

    def dp(i: int, fatigue_vector: tuple) -> float:
        if i == n_games:
            return 0.0

        key = (i, fatigue_vector)
        if key in memo:
            return memo[key][0]

        game = games[i]
        bonus = bonus_results[i]
        eligible = _eligible_pitchers(fatigue_vector, pitchers, game, bonus)
        days_gap = days_gaps[i]

        best_val = NEG_INF
        best_choice = None

        for j in eligible:
            f = fatigue_vector[j]
            score, _ = score_cache[(i, j, f)]

            next_vec = _advance_fatigue(fatigue_vector, j, days_gap)
            future_val = dp(i + 1, next_vec)

            total = score + future_val
            if total > best_val:
                best_val = total
                best_choice = j

        memo[key] = (best_val, best_choice)
        return best_val

    total_score = dp(0, initial_fatigue)

    # ── Traceback ───────────────────────────────────────────────
    assignments = []
    vec = initial_fatigue
    for i in range(n_games):
        _, best_j = memo[(i, vec)]
        f = vec[best_j]
        score, breakdown = score_cache[(i, best_j, f)]

        assignments.append(AssignedStart(
            game            = games[i],
            pitcher         = pitchers[best_j],
            fatigue_level   = f,
            matchup_score   = breakdown["matchup_total"],
            division_bonus  = breakdown["division_bonus"],
            fatigue_penalty = breakdown["fatigue_penalty"],
            total_score     = breakdown["total"],
            score_breakdown = breakdown,
        ))

        vec = _advance_fatigue(vec, best_j, days_gaps[i])

    return OptimizationResult(
        assignments  = assignments,
        total_score  = float(total_score),
        run_date     = run_date,
        horizon_days = (games[-1].date - games[0].date).days if games else 0,
        our_team_id  = our_team_id,
    )