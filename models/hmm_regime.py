"""
HMM Regime Detection — 3-State Gaussian Hidden Markov Model.

Classifies the current market into one of three regimes:
  State 0: Low-Volatility Trend (ATR compressed, directional moves)
  State 1: High-Volatility Choppy (wide ranges, no trend)
  State 2: High-Volatility Crash/Spike (extreme moves, flight-to-safety)

Trained on rolling 2 years of daily returns. Refitted daily with
the latest data. Model serialized to disk for persistence.
"""
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import date, timedelta
from hmmlearn.hmm import GaussianHMM

from models.database import get_connection, fetch_all

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(
    os.path.dirname(__file__), "..", "model_artifacts"
))

# Regime labels mapped by volatility characteristics
REGIME_LABELS = {
    0: "low_vol_trend",
    1: "high_vol_choppy",
    2: "high_vol_crash",
}


class RegimeDetector:
    """3-state Gaussian HMM for market regime classification."""

    def __init__(self, n_states: int = 3, lookback_days: int = 504):
        """
        Args:
            n_states: Number of hidden states (default 3).
            lookback_days: Rolling training window in trading days (~2 years).
        """
        self.n_states = n_states
        self.lookback_days = lookback_days
        self.model: GaussianHMM | None = None
        self.state_map: dict = {}  # Maps model states to semantic labels
        self._model_version: str = ""

    def fit(self, instrument: str = "EUR_USD") -> str:
        """
        Fit the HMM on the most recent 2 years of daily returns.

        Args:
            instrument: Instrument to train on.

        Returns:
            Model version string.
        """
        bars = self._load_training_data(instrument)
        if len(bars) < 60:
            raise ValueError(f"Need at least 60 bars for HMM, got {len(bars)}")

        # Features: daily log returns + 5-day realized volatility
        df = pd.DataFrame(bars)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["vol_5d"] = df["log_return"].rolling(5).std()
        df = df.dropna()

        X = df[["log_return", "vol_5d"]].values

        # Fit Gaussian HMM
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type="full",
            n_iter=200,
            random_state=42,
            tol=1e-4,
        )
        self.model.fit(X)

        # Map states to semantic labels by sorting on mean volatility
        means = self.model.means_
        vol_means = means[:, 1]  # vol_5d column
        sorted_states = np.argsort(vol_means)  # lowest vol first

        self.state_map = {
            int(sorted_states[0]): 0,  # low vol → state 0
            int(sorted_states[1]): 1,  # medium vol → state 1
            int(sorted_states[2]): 2,  # high vol → state 2
        }

        self._model_version = f"hmm_v1_{date.today().isoformat()}"
        self._save_model()

        logger.info(
            f"HMM fitted on {len(X)} samples. State mapping: {self.state_map}. "
            f"Means: {means.tolist()}"
        )
        return self._model_version

    def predict_regime(self, instrument: str = "EUR_USD") -> dict:
        """
        Predict the current regime for today.

        Returns:
            Dict with state_id, state_label, confidence, days_in_regime,
            transition_prob, model_version.
        """
        if self.model is None:
            self._load_model()

        if self.model is None:
            logger.warning("No HMM model available — returning default regime")
            return self._default_regime()

        bars = self._load_training_data(instrument)
        df = pd.DataFrame(bars)
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
        df["vol_5d"] = df["log_return"].rolling(5).std()
        df = df.dropna()

        if len(df) < 10:
            return self._default_regime()

        X = df[["log_return", "vol_5d"]].values

        # Predict states for entire sequence
        raw_states = self.model.predict(X)
        posteriors = self.model.predict_proba(X)

        # Map raw state to semantic state
        current_raw = int(raw_states[-1])
        current_semantic = self.state_map.get(current_raw, 1)
        current_confidence = float(posteriors[-1][current_raw])

        # Count days in current regime
        days_in = 1
        for i in range(len(raw_states) - 2, -1, -1):
            if self.state_map.get(int(raw_states[i]), -1) == current_semantic:
                days_in += 1
            else:
                break

        # Transition probabilities for current state
        trans_row = self.model.transmat_[current_raw].tolist()
        # Remap to semantic ordering
        trans_mapped = {}
        for raw_s, sem_s in self.state_map.items():
            trans_mapped[REGIME_LABELS[sem_s]] = round(trans_row[raw_s], 4)

        result = {
            "state_id": current_semantic,
            "state_label": REGIME_LABELS[current_semantic],
            "confidence": round(current_confidence, 4),
            "days_in_regime": days_in,
            "transition_prob": trans_mapped,
            "model_version": self._model_version,
        }

        logger.info(
            f"Regime: {result['state_label']} (conf={result['confidence']:.3f}, "
            f"days={result['days_in_regime']})"
        )
        return result

    def store_regime(self, run_date: date, instrument: str, regime: dict):
        """Store the regime classification in the database."""
        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO regimes (date, instrument, state_id, state_label,
                        confidence, days_in_regime, transition_prob, model_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, instrument) DO UPDATE SET
                        state_id = EXCLUDED.state_id,
                        state_label = EXCLUDED.state_label,
                        confidence = EXCLUDED.confidence,
                        days_in_regime = EXCLUDED.days_in_regime,
                        transition_prob = EXCLUDED.transition_prob,
                        model_version = EXCLUDED.model_version,
                        created_at = NOW()
                """, (
                    str(run_date), instrument,
                    regime["state_id"], regime["state_label"],
                    regime["confidence"], regime["days_in_regime"],
                    json.dumps(regime["transition_prob"]),
                    regime.get("model_version", ""),
                ))
            conn.commit()
            logger.info(f"Stored regime for {instrument} on {run_date}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing regime: {e}")
            raise
        finally:
            conn.close()

    def _load_training_data(self, instrument: str) -> list[dict]:
        """Load daily bars from DB for training."""
        rows = fetch_all("""
            SELECT bar_time, open, high, low, close, volume
            FROM bars
            WHERE instrument = %s AND granularity = 'D' AND complete = TRUE
            ORDER BY bar_time ASC
        """, (instrument,))

        # Convert Decimal to float
        for row in rows:
            for col in ["open", "high", "low", "close"]:
                row[col] = float(row[col])
            row["volume"] = int(row["volume"])

        return rows

    def _save_model(self):
        """Serialize model to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, "hmm_regime.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "state_map": self.state_map,
                "version": self._model_version,
            }, f)
        logger.info(f"HMM model saved to {path}")

    def _load_model(self):
        """Load model from disk."""
        path = os.path.join(MODEL_DIR, "hmm_regime.pkl")
        if not os.path.exists(path):
            logger.warning(f"No HMM model file at {path}")
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.state_map = data["state_map"]
            self._model_version = data["version"]
        logger.info(f"HMM model loaded: {self._model_version}")

    def _default_regime(self) -> dict:
        """Default regime when model is unavailable."""
        return {
            "state_id": 1,
            "state_label": "high_vol_choppy",
            "confidence": 0.33,
            "days_in_regime": 0,
            "transition_prob": {
                "low_vol_trend": 0.33,
                "high_vol_choppy": 0.34,
                "high_vol_crash": 0.33,
            },
            "model_version": "default",
        }
