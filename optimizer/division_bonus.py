# optimizer/division_bonus.py
import numpy as np
from dataclasses import dataclass, field
from datetime import date
from typing import Optional, cast
from database.models import Team, StandingsSnapshot
from optimizer.matchup_model import _sigmoid


# ─────────────────────────────────────────────────────────────
# DATA CONTRACTS
# ─────────────────────────────────────────────────────────────

@dataclass
class TeamStanding:
    team_id:          int
    team_name:        str
    wins:             int
    losses:           int
    games_behind:     float         # GB from first place (0.0 if leading)
    games_remaining:  int
    run_differential: int
    last_10:          str           # e.g. "7-3"
    home_record:      str
    away_record:      str
    ops:              float
    division:         str
    same_division:    bool = True   # False for out-of-division opponents


@dataclass
class DivisionStandings:
    division:   str
    season:     int
    as_of_date: date
    teams:      list = field(default_factory=list)   # list[TeamStanding]

    def get_team(self, team_id: int) -> Optional[TeamStanding]:
        return next((t for t in self.teams if t.team_id == team_id), None)

    def leader(self) -> TeamStanding:
        return min(self.teams, key=lambda t: t.games_behind)

    def sorted_by_gb(self) -> list:
        return sorted(self.teams, key=lambda t: t.games_behind)


@dataclass
class DivisionBonusResult:
    """
    Returned to dp_engine.py for each opponent-game pairing.
    `bonus_multiplier` is the single value the DP scoring
    function applies. Everything else is for audit logging.
    """
    opponent_team_id:   int
    opponent_name:      str
    games_behind:       float           # opponent's GB
    our_games_behind:   float
    games_remaining:    int
    magic_number:       Optional[int]   # wins to clinch (our team)
    elimination_number: Optional[int]   # losses before we're out
    urgency_score:      float           # raw [0, 1] before scaling
    bonus_multiplier:   float           # final value DP applies
    regime:             str             # "pennant_race" | "eliminated" | "clinched" | "neutral"
    breakdown:          dict            # sub-scores for logging/debugging


# ─────────────────────────────────────────────────────────────
# SEASON CALENDAR HELPERS
# ─────────────────────────────────────────────────────────────

REGULAR_SEASON_GAMES = 162

def season_progress(as_of_date: date, season: int) -> float:
    """
    [0.0, 1.0] — fraction of season elapsed.
    Urgency is always gated by this so May predictions
    don't get inflated September weights.
    """
    season_start = date(season, 4, 1)
    season_end   = date(season, 10, 1)
    total_days   = (season_end - season_start).days
    elapsed      = max((as_of_date - season_start).days, 0)
    return float(np.clip(elapsed / total_days, 0.0, 1.0))

def compute_magic_number(leader_wins: int, opponent_losses: int) -> Optional[int]:
    """
    Wins needed by the leader to clinch.
    163 - leader_wins - trailing_team_losses.
    """
    mn = 163 - leader_wins - opponent_losses
    return int(mn) if mn > 0 else None

def compute_elimination_number(leader_wins: int, team_wins: int, games_rem: int) -> int:
    """
    Losses before a team is mathematically eliminated.
    """
    en = (leader_wins - team_wins) + games_rem + 1
    return max(int(en), 0)


# ─────────────────────────────────────────────────────────────
# REGIME CLASSIFIER
#
# Gates which formula runs. A team 15 GB with 10 to play
# never gets a non-zero bonus regardless of sigmoid output.
# ─────────────────────────────────────────────────────────────

def classify_regime(
    our_gb:           float,
    opponent_gb:      float,
    our_games_rem:    int,
    season_pct:       float,
    we_are_leader:    bool,
) -> str:
    if season_pct < 0.35:
        return "neutral"

    # Opponent is eliminated — no leverage
    if opponent_gb > 12 and our_games_rem < 25:
        return "eliminated"
    if opponent_gb > 20:
        return "eliminated"

    # We've clinched — protect the roster
    if we_are_leader and our_gb == 0.0 and our_games_rem < 10:
        return "clinched"

    # WE ARE LEADING: opponent is within striking distance of us
    # i.e. they could realistically catch us
    if we_are_leader:
        if opponent_gb <= 9.0:
            return "pennant_race"
        if season_pct > 0.65 and opponent_gb <= 12.0:
            return "pennant_race"
        return "neutral"

    # WE ARE TRAILING: opponent is close to us in standings
    if abs(opponent_gb - our_gb) <= 7.0:
        return "pennant_race"
    if season_pct > 0.70 and opponent_gb <= 9.0:
        return "pennant_race"

    return "neutral"


# ─────────────────────────────────────────────────────────────
# URGENCY SCORE
#
# Three sub-signals combined into one [0, 1] value:
#
#   1. gb_urgency          — how close is the GB gap?
#   2. compression         — how few games remain to fix it?
#   3. h2h_leverage        — how many direct matchups are left?
# ─────────────────────────────────────────────────────────────

def compute_urgency_score(
    our_gb:        float,
    opponent_gb:   float,
    our_games_rem: int,
    h2h_remaining: int,
    season_pct:    float,
) -> tuple[float, dict]:

    # Gap between us and opponent in the standings
    # If we lead (our_gb=0), opponent_gb IS their deficit behind us
    # If we trail (our_gb>0), gap = how far they are ahead/behind
    if our_gb == 0.0:
        # We're leading — urgency = how close is this opponent to catching us
        gb_gap = opponent_gb   # positive = they're behind us (good)
        # Urgency HIGH when opponent is close (small gb_gap)
        GB_THRESHOLD = 8.0
        gb_urgency = float(np.clip(1.0 - (gb_gap / GB_THRESHOLD), 0, 1))
    else:
        # We're trailing — urgency = how close are we to them
        gb_gap = opponent_gb - our_gb
        GB_THRESHOLD = 6.0
        gb_urgency = float(np.clip(1.0 - abs(gb_gap) / GB_THRESHOLD, 0, 1))
        if gb_gap < 0:
            gb_urgency = float(np.clip(gb_urgency * 1.30, 0, 1))


    
    # Schedule compression

    compression = _sigmoid(-(our_games_rem - 60), scale=0.08)
    compression = float(compression * season_pct)

    # H2H leverage
    H2H_MAX    = 7
    h2h_score  = float(np.clip(h2h_remaining / H2H_MAX, 0, 1))
    h2h_leverage = gb_urgency * h2h_score * 0.25

    raw = (
        0.45 * gb_urgency
        + 0.40 * compression
        + 0.15 * h2h_score
        + h2h_leverage
    )
    urgency = float(np.clip(raw, 0.0, 1.0))

    breakdown = {
        "gb_gap":        round(opponent_gb if our_gb == 0.0 else (opponent_gb - our_gb), 2),
        "gb_urgency":    round(gb_urgency, 4),
        "compression":   round(compression, 4),
        "h2h_score":     round(h2h_score, 4),
        "h2h_leverage":  round(h2h_leverage, 4),
        "urgency_raw":   round(raw, 4),
        "urgency_final": round(urgency, 4),
    }
    return urgency, breakdown

# ─────────────────────────────────────────────────────────────
# BONUS MULTIPLIER SCALER
#
# Converts raw urgency [0, 1] into a multiplier the DP
# applies to the ops_gap term in compute_score().
#
# Design intent:
#   urgency = 0.0  →  multiplier = 0.0   (no bonus)
#   urgency = 0.5  →  multiplier = 0.8   (moderate race)
#   urgency = 1.0  →  multiplier = 2.0   (must-win series)
#
# The non-linear curve means urgency has to be meaningfully
# high before it actually moves the needle in the DP.
# ─────────────────────────────────────────────────────────────

def scale_to_multiplier(urgency: float, regime: str) -> float:
    """
    Maps urgency score → bonus multiplier for the DP.
    """
    if regime == "eliminated":
        return 0.0
    if regime == "clinched":
        return 0.10    # tiny bonus: still want wins for seeding/HFA
    if regime == "neutral":
        return float(np.clip(urgency * 0.3, 0, 0.3))   # soft early-season signal

    # pennant_race: full non-linear scaling
    # Use a power curve: multiplier = MAX * urgency^0.7
    # (sub-linear at low urgency, aggressive at high urgency)
    MAX_MULTIPLIER = 2.0
    multiplier = MAX_MULTIPLIER * (urgency ** 0.7)
    return float(np.clip(multiplier, 0.0, MAX_MULTIPLIER))


# ─────────────────────────────────────────────────────────────
# MAIN CALCULATOR CLASS
#
# This is what dp_engine.py imports and calls.
# Instantiated once per optimizer run, not per game loop.
# ─────────────────────────────────────────────────────────────

class DivisionBonusCalculator:
    """
    Usage in dp_engine.py:

        calc = DivisionBonusCalculator(our_team_id=our_team_id, db_session=db)
        ...
        bonus = calc.compute(opponent_team_id, as_of_date)
        dp_score += bonus.bonus_multiplier * ops_gap * matchup["ops_suppression"]
    """

    def __init__(self, our_team_id: int, db_session):
        self.our_team_id  = our_team_id
        self.db_session   = db_session
        self._cache: dict = {}   # cache by (opponent_id, date) — standings don't
                                 # change mid-loop, avoid redundant DB hits

    def compute(
        self,
        opponent_team_id: int,
        as_of_date:       date,
        h2h_remaining:    int = 0,
    ) -> DivisionBonusResult:

        cache_key = (opponent_team_id, as_of_date, h2h_remaining)
        if cache_key in self._cache:
            return self._cache[cache_key]

        our_standings = self._load_our_division(as_of_date)
        our_team      = our_standings.get_team(self.our_team_id)

        if our_team is None:
            return self._null_result(opponent_team_id)

        # Opponent may be in a different division — look them up separately.
        # If they're in our division, reuse the already-loaded snapshot date
        # to avoid a second DB round-trip.
        opponent = our_standings.get_team(opponent_team_id)
        if opponent is None:
            opponent = self._load_opponent(
                opponent_team_id,
                as_of_date,
                our_standings.division,
            )

        if opponent is None:
            return self._null_result(opponent_team_id)

        season      = as_of_date.year
        season_pct  = season_progress(as_of_date, season)
        our_gb      = our_team.games_behind
        opp_gb      = opponent.games_behind
        our_rem     = our_team.games_remaining
        leader      = our_standings.leader()
        we_are_lead = (our_team.team_id == leader.team_id)

        # ── Regime ───────────────────────────────────────────
        # Out-of-division opponents are never part of our pennant race —
        # their GB is relative to their own division leader, not ours,
        # so the urgency signals would be meaningless noise.
        if not opponent.same_division:
            regime = "neutral"
        else:
            regime = classify_regime(
                our_gb        = our_gb,
                opponent_gb   = opp_gb,
                our_games_rem = our_rem,
                season_pct    = season_pct,
                we_are_leader = we_are_lead,
            )

        # ── Magic / Elimination numbers ───────────────────────
        magic_num = None
        elim_num  = None
        if we_are_lead:
            magic_num = compute_magic_number(our_team.wins, opponent.losses)
        else:
            elim_num = compute_elimination_number(leader.wins, our_team.wins, our_rem)

        # ── Urgency ───────────────────────────────────────────
        urgency, breakdown = compute_urgency_score(
            our_gb        = our_gb,
            opponent_gb   = opp_gb,
            our_games_rem = our_rem,
            h2h_remaining = h2h_remaining,
            season_pct    = season_pct,
        )

        # ── Bonus multiplier ──────────────────────────────────
        multiplier = scale_to_multiplier(urgency, regime)

        result = DivisionBonusResult(
            opponent_team_id   = opponent_team_id,
            opponent_name      = opponent.team_name,
            games_behind       = opp_gb,
            our_games_behind   = our_gb,
            games_remaining    = our_rem,
            magic_number       = magic_num,
            elimination_number = elim_num,
            urgency_score      = urgency,
            bonus_multiplier   = multiplier,
            regime             = regime,
            breakdown          = {
                **breakdown,
                "regime":      regime,
                "season_pct":  round(season_pct, 3),
                "magic_num":   magic_num,
                "elim_num":    elim_num,
                "multiplier":  round(multiplier, 4),
            }
        )

        self._cache[cache_key] = result
        return result

    def _load_our_division(self, as_of_date: date) -> DivisionStandings:
        """
        Pull the most recent standings snapshot on or before as_of_date
        for our team's entire division.

        Each team's row is fetched independently (most recent <= as_of_date)
        rather than pinning all teams to one shared snapshot_date. Pinning
        caused teams to silently drop out of the division query whenever their
        row was written on a different date than ours (partial syncs, late
        inserts, etc.), making get_team() return None and misclassifying them
        as out-of-division opponents.
        """
        our_snapshot = (
            self.db_session.query(StandingsSnapshot)
            .filter(StandingsSnapshot.snapshot_date <= as_of_date)
            .filter(StandingsSnapshot.team_id == self.our_team_id)
            .order_by(StandingsSnapshot.snapshot_date.desc())
            .first()
        )
        if our_snapshot is None:
            raise ValueError(
                f"No standings snapshot found on or before {as_of_date} "
                f"for team {self.our_team_id}"
            )

        our_division = cast(str, our_snapshot.division)

        # Find all team_ids that have ever appeared in this division
        division_team_ids = (
            self.db_session.query(StandingsSnapshot.team_id)
            .filter(StandingsSnapshot.division == our_division)
            .distinct()
            .all()
        )

        # For each team, grab their own most recent snapshot <= as_of_date
        teams = []
        for (team_id,) in division_team_ids:
            row = (
                self.db_session.query(StandingsSnapshot)
                .filter(StandingsSnapshot.snapshot_date <= as_of_date)
                .filter(StandingsSnapshot.team_id == team_id)
                .order_by(StandingsSnapshot.snapshot_date.desc())
                .first()
            )
            if row is not None:
                teams.append(self._row_to_standing(row))

        return DivisionStandings(
            division   = our_division,
            season     = as_of_date.year,
            as_of_date = as_of_date,
            teams      = teams,
        )

    def _load_opponent(
        self,
        opponent_team_id: int,
        as_of_date: date,
        our_division: str,
    ) -> Optional[TeamStanding]:

        row = (
            self.db_session.query(StandingsSnapshot)
            .filter(StandingsSnapshot.snapshot_date <= as_of_date)
            .filter(StandingsSnapshot.team_id == opponent_team_id)
            .order_by(StandingsSnapshot.snapshot_date.desc())
            .first()
        )

        if row is None:
            return None

        opponent_division = cast(str, row.division)

        return self._row_to_standing(
            row,
            same_division=(opponent_division == our_division),
        )
    @staticmethod
    def _row_to_standing(row: StandingsSnapshot, same_division: bool = True) -> TeamStanding:
        wins   = cast(int,   row.wins)
        losses = cast(int,   row.losses)
        return TeamStanding(
            team_id          = cast(int,   row.team_id),
            team_name        = cast(str,   row.team_name),
            wins             = wins,
            losses           = losses,
            games_behind     = cast(float, row.games_behind),
            games_remaining  = REGULAR_SEASON_GAMES - (wins + losses),
            run_differential = cast(int,   row.run_differential),
            last_10          = cast(str,   row.last_10),
            home_record      = cast(str,   row.home_record),
            away_record      = cast(str,   row.away_record),
            ops              = cast(float, row.ops),
            division         = cast(str,   row.division),
            same_division    = same_division,
        )

    def _null_result(self, opponent_team_id: int) -> DivisionBonusResult:
        """Safe fallback when standings data is missing."""
        return DivisionBonusResult(
            opponent_team_id   = opponent_team_id,
            opponent_name      = "Unknown",
            games_behind       = 0.0,
            our_games_behind   = 0.0,
            games_remaining    = 0,
            magic_number       = None,
            elimination_number = None,
            urgency_score      = 0.0,
            bonus_multiplier   = 0.0,
            regime             = "neutral",
            breakdown          = {"error": "standings_data_missing"},
        )


# ─────────────────────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────────────────────

def _sigmoid(x: float, scale: float = 10.0) -> float:
    return 1.0 / (1.0 + np.exp(-scale * x))