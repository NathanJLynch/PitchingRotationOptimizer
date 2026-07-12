# # ml/predict.py
# import joblib
# import numpy as np
# from dataclasses import dataclass
# from ml.features import build_matchup_features

# @dataclass
# class ScoringCoefficients:
#     """
#     The output of the ML layer.
#     These coefficients replace the hand-tuned α, β, γ
#     in the DP scoring function — now they're matchup-specific.
#     """
#     k_score:        float   # predicted K% (0–1 range)
#     ops_score:      float   # predicted OPS allowed (lower = better pitcher)
#     wpa_estimate:   float   # predicted win probability added
#     confidence:     float   # model agreement score (used to dampen uncertain predictions)


# class PitcherScoringPredictor:
#     def __init__(self):
#         self.k_model   = joblib.load("ml/models/k_pct_model.pkl")
#         self.ops_model = joblib.load("ml/models/ops_model.pkl")
#         self.wpa_model = joblib.load("ml/models/wpa_model.pkl")

#     def predict(self, pitcher_id: int, team_id: int, db_session) -> ScoringCoefficients:
#         X = build_matchup_features(pitcher_id, team_id, db_session)

#         k_pred   = float(self.k_model.predict(X)[0])
#         ops_pred = float(self.ops_model.predict(X)[0])
#         wpa_pred = float(self.wpa_model.predict(X)[0])

#         # Normalize each score to [0, 1] for the DP
#         k_score   = np.clip(k_pred / 0.40, 0, 1)      # 40% K is ~max elite
#         ops_score = np.clip(1 - (ops_pred / 1.0), 0, 1)  # invert: lower OPS = higher score
#         wpa_score = np.clip((wpa_pred + 0.5) / 1.0, 0, 1) # WPA typically [-0.5, 0.5]

#         # Confidence: how much do k_score and ops_score agree with wpa?
#         # High disagreement = uncertain matchup = dampen prediction
#         sub_mean   = (k_score + ops_score) / 2
#         confidence = 1 - abs(sub_mean - wpa_score)

#         return ScoringCoefficients(
#             k_score=k_score,
#             ops_score=ops_score,
#             wpa_estimate=wpa_score,
#             confidence=confidence
#         )

#     def batch_predict(self, pitcher_ids: list, team_id: int, db_session) -> dict:
#         """Predict all pitchers vs. one opponent in one pass — used by the DP."""
#         return {
#             pid: self.predict(pid, team_id, db_session)
#             for pid in pitcher_ids
#         }

# ml/predict.py
import os
import joblib
import numpy as np
from dataclasses import dataclass

@dataclass
class ScoringCoefficients:
    k_score:      float
    ops_score:    float
    wpa_estimate: float
    confidence:   float


class PitcherScoringPredictor:
    """
    Loads trained models if available. If model artifacts don't
    exist yet (no training data / first run), falls back to
    fixed default coefficients so the API can run end-to-end
    for development and testing.

    Once ml/train.py has been run with real historical data,
    drop the .pkl files in ml/models/ and this automatically
    switches to using them.
    """

    MODEL_DIR = os.environ.get("ML_MODEL_DIR", "ml/models")

    def __init__(self):
        self.k_model   = self._try_load("k_pct_model.pkl")
        self.ops_model = self._try_load("ops_model.pkl")
        self.wpa_model = self._try_load("wpa_model.pkl")

        if self.k_model is None:
            print(
                "[PitcherScoringPredictor] No trained models found in "
                f"'{self.MODEL_DIR}/' — using default coefficients. "
                "Run ml/train.py once historical data is available."
            )

    def _try_load(self, filename: str):
        path = os.path.join(self.MODEL_DIR, filename)
        if os.path.exists(path):
            return joblib.load(path)
        return None

    def predict(self, pitcher_id: int, team_id: int, db_session) -> ScoringCoefficients:
        if self.k_model is None:
            return self._default_coefficients(pitcher_id, team_id, db_session)

        from ml.features import build_matchup_features
        X = build_matchup_features(pitcher_id, team_id, db_session)

        k_pred   = float(self.k_model.predict(X)[0])
        ops_pred = float(self.ops_model.predict(X)[0])
        wpa_pred = float(self.wpa_model.predict(X)[0])

        k_score   = np.clip(k_pred / 0.40, 0, 1)
        ops_score = np.clip(1 - (ops_pred / 1.0), 0, 1)
        wpa_score = np.clip((wpa_pred + 0.5) / 1.0, 0, 1)

        sub_mean   = (k_score + ops_score) / 2
        confidence = 1 - abs(sub_mean - wpa_score)

        return ScoringCoefficients(
            k_score=float(k_score),
            ops_score=float(ops_score),
            wpa_estimate=float(wpa_score),
            confidence=float(confidence)
        )

    def _default_coefficients(self, pitcher_id: int, team_id: int, db_session) -> ScoringCoefficients:
        """
        Fallback used when no trained models exist.

        Computes simple ratio-based scores directly from
        pitcher/team stats in the DB rather than ML predictions.
        Confidence is set low (0.4) so dp_engine.py leans more
        on the rule-based matchup_model.py scores instead.
        """
        from database.models import Pitcher, Team

        pitcher = db_session.query(Pitcher).filter_by(id=pitcher_id).first()
        team    = db_session.query(Team).filter_by(id=team_id).first()

        k_pct      = (pitcher.k_pct if pitcher and pitcher.k_pct else 0.22)
        ops_allowed = (pitcher.ops_allowed if pitcher and pitcher.ops_allowed else 0.720)
        team_ops    = (team.ops if team and team.ops else 0.720)

        k_score   = float(np.clip(k_pct / 0.30, 0, 1))
        ops_score = float(np.clip((team_ops - ops_allowed + 0.15) / 0.30, 0, 1))
        wpa_score = float(np.clip((k_score + ops_score) / 2, 0, 1))

        return ScoringCoefficients(
            k_score=k_score,
            ops_score=ops_score,
            wpa_estimate=wpa_score,
            confidence=0.4,   # low confidence — rely more on matchup_model.py
        )

    def batch_predict(self, pitcher_ids: list, team_id: int, db_session) -> dict:
        return {
            pid: self.predict(pid, team_id, db_session)
            for pid in pitcher_ids
        }