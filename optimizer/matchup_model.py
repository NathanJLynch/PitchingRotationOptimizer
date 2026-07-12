# optimizer/matchup_model.py
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple, Optional
from optimizer.fatigue_model import FatigueState
# ─────────────────────────────────────────────────────────────
# DATA CONTRACTS
# These mirror what your database/models.py ORM objects expose.
# ─────────────────────────────────────────────────────────────

@dataclass
class PitcherProfile:
    team_id:             int
    id:                  int
    k_pct:               float  # strikeout rate, e.g. 0.28
    ops_allowed:         float   # career/rolling OPS allowed, e.g. 0.720
    whiff_pct:           float   # swinging strike %, e.g. 0.13
    zone_pct:            float   # % pitches in zone, e.g. 0.47
    chase_pct:           float   # opponent O-swing %, e.g. 0.31
    fb_velo:             float   # fastball velocity mph
    extension:           float   # release extension in feet
    # Splits: ops_allowed broken out by batter handedness
    ops_allowed_vs_rhb:  float
    ops_allowed_vs_lhb:  float
    # Arsenal usage rates (must sum to 1.0)
    fastball_usage:      float
    breaking_usage:      float
    offspeed_usage:      float
    current_fatigue_state: Optional[FatigueState] = None


@dataclass
class OpponentProfile:
    team_id:             int
    ops:                 float   # team OPS, e.g. 0.750
    whiff_rate:          float   # team-wide whiff %, e.g. 0.24
    chase_rate:          float   # team O-swing %, e.g. 0.28
    k_pct:               float   # team strikeout rate
    woba:                float
    xwoba:               float
    hard_hit_pct:        float   # exit velo >= 95mph rate
    # Lineup composition
    rhb_pct:             float   # % of PA from right-handed batters
    lhb_pct:             float   # % of PA from left-handed batters
    # vs. pitch type: how does this team hit each pitch type
    ops_vs_fastball:     float
    ops_vs_breaking:     float
    ops_vs_offspeed:     float


class MatchupWeights(NamedTuple):
    whiff_compat:  float = 0.30
    ops_suppress:  float = 0.45
    platoon:       float = 0.15
    arsenal:       float = 0.10


# ─────────────────────────────────────────────────────────────
# 1. WHIFF COMPATIBILITY
#
# Core question: does the pitcher's swing-and-miss profile
# match well (or poorly) against this team's tendency to whiff?
#
# Power pitchers (high whiff) should face high-whiff teams.
# Contact pitchers (low whiff) should face low-whiff teams.
#
# This is NOT just subtraction. The shape matters:
#   - A 35% whiff pitcher vs. a 30% whiff team = great
#   - A 10% whiff pitcher vs. a 8% whiff team  = fine — it's
#     a contact matchup that suits both profiles
#   - A 10% whiff pitcher vs. a 30% whiff team = mismatch
#     (contact pitcher leaving strikeouts on the table)
# ─────────────────────────────────────────────────────────────

def whiff_compatibility(pitcher: PitcherProfile, opponent: OpponentProfile) -> float:
    """
    Returns a score in [0, 1].
    1.0 = perfect profile alignment, 0.0 = worst mismatch.
    """

    # ── Component A: raw whiff alignment ─────────────────────
    # How well does pitcher's whiff stuff match opponent's whiff tendency?
    # Use geometric mean so BOTH need to be elevated for top scores.
    # A league-average pitcher (~12% whiff) vs. a league-average team
    # (~24% whiff rate) should score around 0.55 — neutral, not penalized.

    LEAGUE_AVG_PITCHER_WHIFF = 0.115   # ~11.5% swinging strike rate
    LEAGUE_AVG_TEAM_WHIFF    = 0.240   # ~24% whiff rate (swings that miss)

    pitcher_whiff_rel = pitcher.whiff_pct / LEAGUE_AVG_PITCHER_WHIFF   # >1 = above avg
    team_whiff_rel    = opponent.whiff_rate / LEAGUE_AVG_TEAM_WHIFF

    # Geometric mean rewards mutual elevation, penalizes mismatch
    raw_alignment = np.sqrt(pitcher_whiff_rel * team_whiff_rel)
    raw_alignment = np.clip(raw_alignment, 0.5, 2.0)   # cap extremes

    # Normalize to [0, 1]: 0.5 → 0.0, 1.0 → 0.5, 2.0 → 1.0
    alignment_score = (raw_alignment - 0.5) / 1.5

    # ── Component B: chase synergy ────────────────────────────
    # A pitcher with elite chase-inducing ability (high o-swing%)
    # paired with a free-swinging team = extra upside.
    # This captures what whiff% alone misses: the *sequence* of
    # expanding the zone after establishing it.

    chase_synergy = pitcher.chase_pct * opponent.chase_rate
    # Normalize: elite-elite combo (~0.31 * 0.34 ≈ 0.11) → 1.0
    # League avg combo (~0.31 * 0.29 ≈ 0.09) → ~0.55
    ELITE_CHASE_PRODUCT = 0.11
    chase_score = np.clip(chase_synergy / ELITE_CHASE_PRODUCT, 0, 1)

    # ── Component C: zone command bonus ──────────────────────
    # A pitcher who pounds the zone (high zone%) rewards when facing
    # a team with poor plate discipline (high swing%, low walk rate).
    # This slightly boosts contact pitchers in the right matchup.

    zone_command_score = pitcher.zone_pct * (1 - opponent.chase_rate)
    # High zone% + low chase rate team = contact outs, not walks
    zone_command_score = np.clip(zone_command_score / 0.35, 0, 1)

    return float(
        0.60 * alignment_score
        + 0.30 * chase_score
        + 0.10 * zone_command_score
    )


# ─────────────────────────────────────────────────────────────
# 2. OPS SUPPRESSION SCORE
#
# How likely is this pitcher to suppress THIS team's offense?
#
# Raw ops_allowed is a blunt instrument. A pitcher's OPS allowed
# against a .680 OPS team looks different than against a .820 team.
# We want to capture: is this pitcher *better or worse* than what
# this team typically does to pitchers?
# ─────────────────────────────────────────────────────────────

def ops_suppression_score(pitcher: PitcherProfile, opponent: OpponentProfile) -> float:
    """
    Returns a score in [0, 1].
    1.0 = pitcher strongly suppresses this lineup.
    0.0 = lineup is likely to punish this pitcher.
    """

    # ── Component A: expected vs. actual OPS delta ────────────
    # If a team's OPS is .780 but this pitcher only allows .680,
    # that's a +.100 suppression edge. Scale it.

    ops_delta = opponent.ops - pitcher.ops_allowed
    # ops_delta > 0: pitcher suppresses below team's norm (good)
    # ops_delta < 0: pitcher allows more than team's norm (bad)

    # Sigmoid-scale the delta: maps [-0.2, +0.2] to roughly [0.1, 0.9]
    # This prevents extreme outliers from dominating
    ops_delta_score = _sigmoid(ops_delta, scale=15)   # scale=15 → smooth curve

    # ── Component B: xwOBA adjustment ────────────────────────
    # OPS can be noisy (BABIP variance). Weight xwOBA more heavily
    # for pitchers with fewer than ~100 IP, since their OPS allowed
    # is less stabilized. xwOBA regresses faster.

    # We use the team's xwOBA (expected, batted-ball based) as
    # a proxy for "true" offensive quality, ignoring luck.
    LEAGUE_AVG_XWOBA   = 0.315
    LEAGUE_AVG_OPS     = 0.720

    # Rescale xwOBA to OPS-like units for apples-to-apples comparison
    xwoba_adj_ops = LEAGUE_AVG_OPS + (opponent.xwoba - LEAGUE_AVG_XWOBA) * 2.5
    xwoba_delta   = xwoba_adj_ops - pitcher.ops_allowed
    xwoba_score   = _sigmoid(xwoba_delta, scale=15)

    # ── Component C: hard-hit rate suppression ────────────────
    # A high ops_allowed pitcher who keeps the ball on the ground
    # can still be effective vs. a team that hits the ball hard.
    # Hard-hit% is exit-velo based — predictive of true power damage.

    # Pitcher velocity as proxy for hard-hit suppression
    # Elite velo (95+ mph) correlates with suppressing hard contact
    ELITE_VELO = 95.0
    velo_factor = np.clip(pitcher.fb_velo / ELITE_VELO, 0.85, 1.10)

    # Team hard-hit tendency: punishes soft-tossers more
    hard_hit_exposure = opponent.hard_hit_pct * (2.0 - velo_factor)
    # hard_hit_exposure high = pitcher gets hurt by hard-hit team
    hard_hit_score = np.clip(1.0 - (hard_hit_exposure / 0.50), 0, 1)

    return float(
        0.50 * ops_delta_score
        + 0.35 * xwoba_score
        + 0.15 * hard_hit_score
    )


# ─────────────────────────────────────────────────────────────
# 3. PLATOON ADJUSTMENT
#
# Does the lineup's handedness composition favor or hurt
# this pitcher's known L/R splits?
#
# A RHP with a big reverse split (better vs. LHB) facing a
# lefty-heavy lineup should score higher, not lower.
# ─────────────────────────────────────────────────────────────

def platoon_adjustment(pitcher: PitcherProfile, opponent: OpponentProfile) -> float:
    """
    Returns a score in [0, 1].
    Reflects how well pitcher's splits align with lineup composition.
    """

    # ── Step 1: compute pitcher's platoon advantage direction ─
    # Positive = pitcher is better vs RHB (normal for LHP)
    # Negative = pitcher is better vs LHB (normal for RHP, or reverse split)

    rhb_edge = pitcher.ops_allowed_vs_lhb - pitcher.ops_allowed_vs_rhb
    # If ops_allowed_vs_lhb > ops_allowed_vs_rhb → pitcher is WORSE vs LHB
    # Positive rhb_edge means pitcher favors RHB matchups

    lhb_edge = pitcher.ops_allowed_vs_rhb - pitcher.ops_allowed_vs_lhb
    # Positive lhb_edge means pitcher favors LHB matchups

    # ── Step 2: weight edges by lineup composition ────────────
    # A .040 platoon split matters a lot vs. an 80% same-handed lineup
    # but barely matters vs. a balanced 50/50 lineup

    weighted_advantage = (
        rhb_edge * opponent.rhb_pct   # pitcher's RHB edge × lineup's RHB share
        + lhb_edge * opponent.lhb_pct  # pitcher's LHB edge × lineup's LHB share
    )
    # weighted_advantage > 0: lineup composition favors this pitcher
    # weighted_advantage < 0: lineup composition hurts this pitcher

    # ── Step 3: normalize to [0, 1] ──────────────────────────
    # Typical platoon splits are ±.040 OPS. A full favorable lineup
    # (e.g., 80% same-handed) → max advantage ~.032.
    MAX_PLATOON_ADVANTAGE = 0.032
    platoon_score = 0.5 + (weighted_advantage / (2 * MAX_PLATOON_ADVANTAGE))
    platoon_score = np.clip(platoon_score, 0, 1)

    # ── Step 4: arsenal interaction ──────────────────────────
    # A pitcher whose arsenal is heavily breaking-ball based has
    # larger platoon splits. A four-seam/changeup pitcher has less.
    # Scale the platoon signal by how much the arsenal CREATES splits.

    # Breaking balls create larger splits; changeups create reverse splits
    split_amplifier = 1.0 + (pitcher.breaking_usage - pitcher.offspeed_usage) * 0.5
    split_amplifier = np.clip(split_amplifier, 0.7, 1.3)

    amplified = 0.5 + (platoon_score - 0.5) * split_amplifier
    return float(np.clip(amplified, 0, 1))


# ─────────────────────────────────────────────────────────────
# 4. ARSENAL MATCHUP SCORE
#
# How does this pitcher's pitch mix interact with what this
# team is good (or bad) at hitting?
#
# A team that murders fastballs but can't touch breaking balls
# should push the score down for a four-seam heavy pitcher.
# ─────────────────────────────────────────────────────────────

def arsenal_matchup_score(pitcher: PitcherProfile, opponent: OpponentProfile) -> float:
    """
    Returns a score in [0, 1].
    High = pitcher's pitch mix exploits team's weaknesses.
    """

    # For each pitch type: compute (team weakness) × (pitcher usage)
    # Team weakness = how poorly they hit that pitch type
    # Usage = how often pitcher throws it → exposure-weighted damage

    LEAGUE_AVG_OPS_VS_FB      = 0.760
    LEAGUE_AVG_OPS_VS_BREAKING = 0.670
    LEAGUE_AVG_OPS_VS_OFFSPEED = 0.700

    # Weakness scores: higher = team is worse vs. that pitch type
    fb_weakness       = np.clip((LEAGUE_AVG_OPS_VS_FB - opponent.ops_vs_fastball) / 0.15, -1, 1)
    breaking_weakness = np.clip((LEAGUE_AVG_OPS_VS_BREAKING - opponent.ops_vs_breaking) / 0.15, -1, 1)
    offspeed_weakness = np.clip((LEAGUE_AVG_OPS_VS_OFFSPEED - opponent.ops_vs_offspeed) / 0.15, -1, 1)

    # Weighted by pitcher's arsenal usage
    raw_score = (
        pitcher.fastball_usage  * fb_weakness
        + pitcher.breaking_usage  * breaking_weakness
        + pitcher.offspeed_usage  * offspeed_weakness
    )

    # raw_score ∈ [-1, 1] → normalize to [0, 1]
    return float(np.clip((raw_score + 1) / 2, 0, 1))


# ─────────────────────────────────────────────────────────────
# 5. MASTER SCORING FUNCTION
# ─────────────────────────────────────────────────────────────

def compute_matchup_score(
    pitcher:  PitcherProfile,
    opponent: OpponentProfile,
    weights:  MatchupWeights = MatchupWeights(),
    ml_coefficients: Optional[dict] = None,   # injected from ml/predict.py at runtime
) -> dict:
    """
    Master matchup score. Returns a dict so the DP can inspect
    sub-scores individually (useful for debugging and logging).

    ml_coefficients: if provided, overrides the static weights
    with model-predicted context-aware values. Shape:
      {
        "whiff_weight": float,
        "ops_weight":   float,
        "platoon_weight": float,
        "arsenal_weight": float,
      }
    """

    # ── Compute sub-scores ────────────────────────────────────
    whiff   = whiff_compatibility(pitcher, opponent)
    ops     = ops_suppression_score(pitcher, opponent)
    platoon = platoon_adjustment(pitcher, opponent)
    arsenal = arsenal_matchup_score(pitcher, opponent)

    # ── Resolve weights ───────────────────────────────────────
    if ml_coefficients:
        # ML layer overrides static defaults with matchup-specific weights
        w_whiff   = ml_coefficients.get("whiff_weight",   weights.whiff_compat)
        w_ops     = ml_coefficients.get("ops_weight",     weights.ops_suppress)
        w_platoon = ml_coefficients.get("platoon_weight", weights.platoon)
        w_arsenal = ml_coefficients.get("arsenal_weight", weights.arsenal)

        # Re-normalize so weights always sum to 1.0
        total = w_whiff + w_ops + w_platoon + w_arsenal
        w_whiff, w_ops, w_platoon, w_arsenal = (
            w_whiff/total, w_ops/total, w_platoon/total, w_arsenal/total
        )
    else:
        w_whiff   = weights.whiff_compat
        w_ops     = weights.ops_suppress
        w_platoon = weights.platoon
        w_arsenal = weights.arsenal

    # ── Weighted sum ──────────────────────────────────────────
    final = (
        w_whiff   * whiff
        + w_ops     * ops
        + w_platoon * platoon
        + w_arsenal * arsenal
    )

    return {
        "total":          round(float(final), 4),
        "whiff_compat":   round(whiff,   4),
        "ops_suppression": round(ops,    4),
        "platoon":        round(platoon, 4),
        "arsenal":        round(arsenal, 4),
        "weights_used":   {
            "whiff": w_whiff, "ops": w_ops,
            "platoon": w_platoon, "arsenal": w_arsenal,
        }
    }


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────

def _sigmoid(x: float, scale: float = 10.0) -> float:
    """Smooth S-curve mapping any real number to (0, 1)."""
    return 1.0 / (1.0 + np.exp(-scale * x))