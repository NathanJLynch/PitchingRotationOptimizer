from datetime import date
from typing import Any
from unittest.mock import patch

import pytest

from optimizer.dp_engine import solve, Game
from optimizer.matchup_model import PitcherProfile, OpponentProfile, compute_matchup_score
from optimizer.division_bonus import DivisionBonusResult
from ml.predict import ScoringCoefficients


def make_pitcher(id, k_pct, ops_allowed, whiff_pct=0.13, **overrides):
    defaults: dict[str, Any] = dict(
        team_id=1,
        zone_pct=0.47,
        chase_pct=0.30,
        fb_velo=93.0,
        extension=6.2,
        ops_allowed_vs_rhb=ops_allowed,
        ops_allowed_vs_lhb=ops_allowed,
        fastball_usage=0.5,
        breaking_usage=0.3,
        offspeed_usage=0.2,
    )
    defaults.update(overrides)
    return PitcherProfile(id=id, k_pct=k_pct, ops_allowed=ops_allowed, whiff_pct=whiff_pct, **defaults)


def make_opponent(**overrides):
    defaults: dict[str, Any] = dict(
        team_id=99,
        ops=0.780,
        whiff_rate=0.32,
        chase_rate=0.30,
        k_pct=0.26,
        woba=0.330,
        xwoba=0.330,
        hard_hit_pct=0.40,
        rhb_pct=0.6,
        lhb_pct=0.4,
        ops_vs_fastball=0.760,
        ops_vs_breaking=0.670,
        ops_vs_offspeed=0.700,
    )
    defaults.update(overrides)
    return OpponentProfile(**defaults)


def make_game(opponent, game_id=1, d=date(2026, 4, 1), **overrides):
    defaults: dict[str, Any] = dict(
        game_id=game_id,
        date=d,
        opponent_team_id=opponent.team_id,
        opponent=opponent,
        is_home=True,
        series_game_num=1,
        h2h_remaining=3,
    )
    defaults.update(overrides)
    return Game(**defaults)


def make_no_bonus_result(opponent_team_id: int = 99) -> DivisionBonusResult:
    """
    A neutral division-bonus result: regime='none' so dp_engine.compute_score's
    pennant_race/clinched branches never fire. Every other field is a type-valid
    placeholder -- dp_engine only reads .regime (branching) and .urgency_score
    (logged into the breakdown dict), so nothing else here affects test outcomes.
    """
    return DivisionBonusResult(
        opponent_team_id=opponent_team_id,
        opponent_name="Test Opponent",
        games_behind=0.0,
        our_games_behind=0.0,
        games_remaining=0,
        magic_number=None,
        elimination_number=None,
        urgency_score=0.0,
        bonus_multiplier=0.0,
        regime="none",
        breakdown={},
    )


@pytest.fixture(autouse=True)
def mock_division_bonus():
    """
    Patches DivisionBonusCalculator everywhere dp_engine.py uses it, so solve()
    never touches db_session (which these tests pass as None). Patching at
    'optimizer.dp_engine.DivisionBonusCalculator' -- the imported reference
    dp_engine.py actually calls -- not at 'optimizer.division_bonus', which
    would leave dp_engine's already-imported reference untouched.
    """
    with patch("optimizer.dp_engine.DivisionBonusCalculator") as mock_calc_cls:
        mock_calc_cls.return_value.compute.return_value = make_no_bonus_result()
        yield mock_calc_cls


def test_high_k_confidence_favors_whiff_heavy_pitcher_over_ops_pitcher():
    """
    Pitcher A: elite whiff stuff, mediocre OPS suppression.
    Pitcher B: mediocre whiff stuff, elite OPS suppression.
    Opponent: a high-whiff-rate team (favors A on whiff_compat).

    If ML predicts high k_score/confidence for A vs this opponent, the DP should
    prefer A more strongly than it would under static default weights alone.
    """
    pitcher_a = make_pitcher(id=1, k_pct=0.32, ops_allowed=0.740, whiff_pct=0.30)
    pitcher_b = make_pitcher(id=2, k_pct=0.18, ops_allowed=0.620, whiff_pct=0.10)
    opponent = make_opponent()
    game = make_game(opponent)

    def fake_predict(self, pitcher_id, team_id, db_session):
        if pitcher_id == 1:
            return ScoringCoefficients(k_score=0.95, ops_score=0.2, wpa_estimate=0.6, confidence=0.9)
        return ScoringCoefficients(k_score=0.3, ops_score=0.9, wpa_estimate=0.6, confidence=0.9)

    with patch("ml.predict.PitcherScoringPredictor.predict", fake_predict):
        result = solve(games=[game], pitchers=[pitcher_a, pitcher_b], our_team_id=1, db_session=None)

    assert result.assignments[0].pitcher.id == 1, (
        "With ML strongly favoring pitcher A's whiff dimension against a high-whiff "
        "opponent, the DP should select A over B."
    )


def test_zero_confidence_dp_matches_static_only_choice():
    """
    Same two-pitcher setup, but with confidence forced to 0 for both pitchers.
    This isolates 'did the ML weighting change the decision' from 'did the
    static formula alone determine this' -- the choice here should match
    whatever the DP would pick with ml_coefficients=None throughout.
    """
    pitcher_a = make_pitcher(id=1, k_pct=0.32, ops_allowed=0.740, whiff_pct=0.30)
    pitcher_b = make_pitcher(id=2, k_pct=0.18, ops_allowed=0.620, whiff_pct=0.10)
    opponent = make_opponent()
    game = make_game(opponent)

    def fake_predict_zero_conf(self, pitcher_id, team_id, db_session):
        return ScoringCoefficients(k_score=0.95 if pitcher_id == 1 else 0.3,
                                    ops_score=0.2 if pitcher_id == 1 else 0.9,
                                    wpa_estimate=0.6, confidence=0.0)

    with patch("ml.predict.PitcherScoringPredictor.predict", fake_predict_zero_conf):
        result_zero_conf = solve(games=[game], pitchers=[pitcher_a, pitcher_b], our_team_id=1, db_session=None)

    # Reconstruct the same scoring the DP would do with ml_coefficients=None.
    # Division bonus is neutralized by the autouse fixture and fatigue is equal
    # for both pitchers on game 1, so the matchup score alone should decide it.
    score_a = compute_matchup_score(pitcher_a, opponent, ml_coefficients=None)["total"]
    score_b = compute_matchup_score(pitcher_b, opponent, ml_coefficients=None)["total"]
    expected_pitcher_id = pitcher_a.id if score_a >= score_b else pitcher_b.id

    assert result_zero_conf.assignments[0].pitcher.id == expected_pitcher_id, (
        "At confidence=0, the DP's choice should match the static-weights-only "
        "matchup comparison (division bonus / fatigue held equal here)."
    )


def test_ml_weighting_does_not_break_multi_game_solve():
    """Smoke test: a short multi-game horizon with ML weighting enabled should
    solve without error and produce one assignment per game."""
    pitcher_a = make_pitcher(id=1, k_pct=0.30, ops_allowed=0.700)
    pitcher_b = make_pitcher(id=2, k_pct=0.22, ops_allowed=0.680)
    opponent = make_opponent()

    games = [
        make_game(opponent, game_id=1, d=date(2026, 4, 1)),
        make_game(opponent, game_id=2, d=date(2026, 4, 2)),
        make_game(opponent, game_id=3, d=date(2026, 4, 5)),
    ]

    def fake_predict(self, pitcher_id, team_id, db_session):
        return ScoringCoefficients(k_score=0.6, ops_score=0.5, wpa_estimate=0.5, confidence=0.7)

    with patch("ml.predict.PitcherScoringPredictor.predict", fake_predict):
        result = solve(games=games, pitchers=[pitcher_a, pitcher_b], our_team_id=1, db_session=None)

    assert len(result.assignments) == len(games)
    assert all(a.pitcher.id in (1, 2) for a in result.assignments)