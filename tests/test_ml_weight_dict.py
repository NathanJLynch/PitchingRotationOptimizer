from typing import Any

import pytest
import numpy as np

from optimizer.matchup_model import (
    build_ml_weight_dict,
    compute_matchup_score,
    MatchupWeights,
    PitcherProfile,
    OpponentProfile,
)
from ml.predict import ScoringCoefficients

BASE = MatchupWeights()  # whiff=0.30, ops=0.45, platoon=0.15, arsenal=0.10


def make_coeffs(k=0.5, ops=0.5, wpa=0.5, confidence=0.5):
    return ScoringCoefficients(k_score=k, ops_score=ops, wpa_estimate=wpa, confidence=confidence)


def make_pitcher(**overrides):
    defaults: dict[str, Any] = dict(
        team_id=1,
        id=1,
        k_pct=0.25,
        ops_allowed=0.700,
        whiff_pct=0.13,
        zone_pct=0.47,
        chase_pct=0.30,
        fb_velo=93.0,
        extension=6.2,
        ops_allowed_vs_rhb=0.700,
        ops_allowed_vs_lhb=0.700,
        fastball_usage=0.5,
        breaking_usage=0.3,
        offspeed_usage=0.2,
    )
    defaults.update(overrides)
    return PitcherProfile(**defaults)


def make_opponent(**overrides):
    defaults: dict[str, Any] = dict(
        team_id=99,
        ops=0.750,
        whiff_rate=0.25,
        chase_rate=0.29,
        k_pct=0.22,
        woba=0.320,
        xwoba=0.315,
        hard_hit_pct=0.37,
        rhb_pct=0.5,
        lhb_pct=0.5,
        ops_vs_fastball=0.760,
        ops_vs_breaking=0.670,
        ops_vs_offspeed=0.700,
    )
    defaults.update(overrides)
    return OpponentProfile(**defaults)


# ─────────────────────────────────────────────────────────────
# build_ml_weight_dict unit tests
# ─────────────────────────────────────────────────────────────

def test_zero_confidence_returns_static_defaults():
    """confidence=0 should be a pure passthrough of base_weights, regardless of scores."""
    coeffs = make_coeffs(k=0.9, ops=0.1, confidence=0.0)
    w = build_ml_weight_dict(coeffs, BASE)
    assert w["whiff_weight"] == pytest.approx(BASE.whiff_compat)
    assert w["ops_weight"] == pytest.approx(BASE.ops_suppress)
    assert w["platoon_weight"] == pytest.approx(BASE.platoon)
    assert w["arsenal_weight"] == pytest.approx(BASE.arsenal)


def test_platoon_and_arsenal_never_move():
    """Nothing in ScoringCoefficients models these -- they must stay fixed at any confidence."""
    for conf in [0.0, 0.3, 0.7, 1.0]:
        coeffs = make_coeffs(k=0.9, ops=0.1, confidence=conf)
        w = build_ml_weight_dict(coeffs, BASE)
        assert w["platoon_weight"] == pytest.approx(BASE.platoon)
        assert w["arsenal_weight"] == pytest.approx(BASE.arsenal)


def test_whiff_ops_budget_is_conserved():
    """whiff_weight + ops_weight should always equal the static whiff+ops budget,
    regardless of confidence or k/ops split -- only the split between them should move."""
    budget = BASE.whiff_compat + BASE.ops_suppress
    for conf in [0.0, 0.25, 0.5, 0.75, 1.0]:
        for k, ops in [(0.9, 0.1), (0.1, 0.9), (0.5, 0.5), (0.0, 0.0)]:
            coeffs = make_coeffs(k=k, ops=ops, confidence=conf)
            w = build_ml_weight_dict(coeffs, BASE)
            assert w["whiff_weight"] + w["ops_weight"] == pytest.approx(budget, abs=1e-9)


def test_full_confidence_follows_relative_emphasis():
    """confidence=1.0, k_score >> ops_score -> whiff_weight should dominate the budget."""
    coeffs = make_coeffs(k=0.9, ops=0.1, confidence=1.0)
    w = build_ml_weight_dict(coeffs, BASE)
    budget = BASE.whiff_compat + BASE.ops_suppress
    assert w["whiff_weight"] == pytest.approx(0.9 * budget, abs=1e-6)
    assert w["ops_weight"] == pytest.approx(0.1 * budget, abs=1e-6)


def test_zero_k_and_ops_score_falls_back_to_even_split():
    """Guards the total > 1e-6 branch -- no signal should mean no redistribution bias."""
    coeffs = make_coeffs(k=0.0, ops=0.0, confidence=1.0)
    w = build_ml_weight_dict(coeffs, BASE)
    budget = BASE.whiff_compat + BASE.ops_suppress
    assert w["whiff_weight"] == pytest.approx(budget / 2, abs=1e-6)
    assert w["ops_weight"] == pytest.approx(budget / 2, abs=1e-6)


def test_weights_are_monotonic_in_confidence():
    """As confidence rises from 0 to 1 with a fixed k/ops split, whiff_weight should
    move monotonically toward the ML-implied value (not oscillate)."""
    coeffs_lo_to_hi = [make_coeffs(k=0.9, ops=0.1, confidence=c) for c in np.linspace(0, 1, 11)]
    whiff_weights = [build_ml_weight_dict(c, BASE)["whiff_weight"] for c in coeffs_lo_to_hi]
    diffs = np.diff(whiff_weights)
    assert all(d >= -1e-9 for d in diffs), (
        "whiff_weight should increase monotonically as confidence rises toward a "
        "k-favoring signal"
    )


def test_output_weights_sum_to_static_total():
    """Total weight budget across all four dims should equal base_weights total (1.0),
    since compute_matchup_score re-normalizes anyway, but this catches silent drift."""
    coeffs = make_coeffs(k=0.7, ops=0.3, confidence=0.6)
    w = build_ml_weight_dict(coeffs, BASE)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("confidence", [-0.5, 1.5])
def test_out_of_range_confidence_is_clipped(confidence):
    """confidence should theoretically stay in [0,1] but guard against upstream bugs."""
    coeffs = make_coeffs(k=0.9, ops=0.1, confidence=confidence)
    w = build_ml_weight_dict(coeffs, BASE)
    budget = BASE.whiff_compat + BASE.ops_suppress
    assert 0 <= w["whiff_weight"] <= budget
    assert 0 <= w["ops_weight"] <= budget


# ─────────────────────────────────────────────────────────────
# One layer up: does using the derived weights actually behave
# as a no-op at confidence=0 once plugged into compute_matchup_score?
# (Same invariant as test_zero_confidence_returns_static_defaults,
# but exercised through the actual consumer instead of just the dict.)
# ─────────────────────────────────────────────────────────────

def test_zero_confidence_matches_no_ml_matchup_score():
    """The final matchup score using ml_coefficients at confidence=0 must exactly
    match calling compute_matchup_score with ml_coefficients=None."""
    pitcher = make_pitcher()
    opponent = make_opponent()

    coeffs = make_coeffs(k=0.9, ops=0.1, confidence=0.0)
    ml_weights = build_ml_weight_dict(coeffs)

    with_ml = compute_matchup_score(pitcher, opponent, ml_coefficients=ml_weights)
    without_ml = compute_matchup_score(pitcher, opponent, ml_coefficients=None)

    assert with_ml["total"] == pytest.approx(without_ml["total"], abs=1e-9)
    assert with_ml["weights_used"] == pytest.approx(without_ml["weights_used"], abs=1e-9)