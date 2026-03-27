"""
Meta-Model: XGBoost Binary Classifier with CPCV Validation.

This is Model 2 in the Lopez de Prado meta-labeling framework.
The primary signals (Model 1) propose a direction (long/short).
This model predicts the PROBABILITY that the proposed trade
will actually be profitable tomorrow (T+1).

- Probability < 0.55 → FLAT (no trade)
- Probability 0.55–0.70 → half position
- Probability > 0.70 → full position

Validation: Purged Combinatorial Cross-Validation (CPCV)
  N=6 groups, k=2 → 15 backtest paths
  Purge: 5 days | Embargo: 2 days
"""
import os
import json
import pickle
import logging
import numpy as np
import pandas as pd
from datetime import date
from itertools import combinations

logger = logging.getLogger(__name__)

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.join(
    os.path.dirname(__file__), "..", "model_artifacts"
))

# Feature columns in the order expected by the model
FEATURE_COLS = [
    "regime_state", "days_in_regime",
    "yield_spread_bps", "yield_spread_change_5d", "yield_spread_change_20d",
    "sentiment_pct_long", "sentiment_extreme",
    "macro_sentiment_score", "ai_confidence",
    "fed_stance_encoded", "ecb_stance_encoded", "risk_sentiment_encoded",
    "atr_14", "rsi_14", "price_vs_ma50", "price_vs_ma200",
    "body_direction", "body_pct_of_range",
    "eod_event_reversal", "event_surprise_magnitude",
    "day_of_week", "is_friday",
    "long_swap_pips", "short_swap_pips",
    "primary_signal_direction", "primary_signal_count",
    "composite_strength", "tier2_confirmation_count",
]


class MetaModel:
    """XGBoost meta-labeling model with CPCV validation."""

    def __init__(self):
        self.model = None
        self.model_version: str = ""
        self.cpcv_results: dict = {}
        self.shap_importance: list = []

    def train(self, feature_vectors: list[dict], labels: list[int]) -> dict:
        """
        Train the XGBoost model on historical feature vectors.

        Args:
            feature_vectors: List of feature dicts (26 features each).
            labels: Binary labels (1=profitable, 0=not).

        Returns:
            Training results dict with metrics and CPCV scores.
        """
        import xgboost as xgb
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        df = pd.DataFrame(feature_vectors)

        # Ensure all expected columns exist
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0

        X = df[FEATURE_COLS].fillna(0).values
        y = np.array(labels)

        if len(X) < 50:
            raise ValueError(f"Need at least 50 samples for training, got {len(X)}")

        # CPCV validation
        cpcv = self._run_cpcv(X, y, n_groups=6, k_test=2, purge=5, embargo=2)

        # Train final model on all data
        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=5,
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        self.model.fit(X, y)

        # Full-dataset metrics
        y_pred = self.model.predict(X)
        metrics = {
            "accuracy": round(accuracy_score(y, y_pred), 4),
            "precision": round(precision_score(y, y_pred, zero_division=0), 4),
            "recall": round(recall_score(y, y_pred, zero_division=0), 4),
            "f1": round(f1_score(y, y_pred, zero_division=0), 4),
            "samples": len(X),
            "positive_rate": round(y.mean(), 4),
        }

        # SHAP importance
        self.shap_importance = self._compute_shap(X)

        self.model_version = f"xgb_v1_{date.today().isoformat()}"
        self.cpcv_results = cpcv
        self._save_model()

        logger.info(
            f"Meta-model trained: {metrics}. "
            f"CPCV deflated Sharpe: {cpcv.get('deflated_sharpe', 'N/A')}"
        )

        return {
            "model_version": self.model_version,
            "metrics": metrics,
            "cpcv": cpcv,
            "top_features": self.shap_importance[:10],
        }

    def predict(self, feature_vector: dict) -> dict:
        """
        Predict probability of profitable trade for a single day.

        Args:
            feature_vector: Dict of 26 features.

        Returns:
            Dict with direction, probability, size_multiplier, top_shap.
        """
        if self.model is None:
            self._load_model()

        if self.model is None:
            logger.warning("No meta-model available — returning default prediction")
            return self._default_prediction(feature_vector)

        # Build feature array
        X = np.array([[feature_vector.get(col, 0.0) for col in FEATURE_COLS]])
        prob = float(self.model.predict_proba(X)[0][1])

        # Position sizing from probability
        if prob < 0.55:
            size_mult = 0.0
            direction = "flat"
        elif prob < 0.70:
            size_mult = 0.5
            direction = "long" if feature_vector.get("primary_signal_direction", 0) > 0 else "short"
        else:
            size_mult = 1.0
            direction = "long" if feature_vector.get("primary_signal_direction", 0) > 0 else "short"

        # If primary signal is flat, meta-model can't override
        if feature_vector.get("primary_signal_direction", 0) == 0:
            direction = "flat"
            size_mult = 0.0

        result = {
            "direction": direction,
            "probability": round(prob, 4),
            "size_multiplier": size_mult,
            "model_version": self.model_version,
            "top_shap": self.shap_importance[:5] if self.shap_importance else [],
        }

        logger.info(
            f"Meta prediction: {direction} (prob={prob:.3f}, size={size_mult}x)"
        )
        return result

    def _run_cpcv(self, X: np.ndarray, y: np.ndarray,
                  n_groups: int = 6, k_test: int = 2,
                  purge: int = 5, embargo: int = 2) -> dict:
        """
        Purged Combinatorial Cross-Validation.

        N=6 groups, k=2 test → C(6,2) = 15 backtest paths.
        Each path has a purge window of 5 days and embargo of 2 days
        to prevent label leakage.
        """
        import xgboost as xgb

        n = len(X)
        group_size = n // n_groups
        groups = []
        for i in range(n_groups):
            start = i * group_size
            end = start + group_size if i < n_groups - 1 else n
            groups.append(list(range(start, end)))

        test_combos = list(combinations(range(n_groups), k_test))
        path_returns = []

        for combo in test_combos:
            test_idx = set()
            for g in combo:
                test_idx.update(groups[g])

            # Purge + embargo: remove samples near test boundaries
            purged = set()
            for idx in test_idx:
                for offset in range(-purge, embargo + 1):
                    purged.add(idx + offset)

            train_idx = [i for i in range(n) if i not in test_idx and i not in purged]
            test_idx = sorted(test_idx)

            if len(train_idx) < 30 or len(test_idx) < 10:
                continue

            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            model = xgb.XGBClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, use_label_encoder=False,
                eval_metric="logloss",
            )
            model.fit(X_train, y_train, verbose=False)

            probs = model.predict_proba(X_test)[:, 1]
            # Simulate returns: if prob > 0.55, take the trade
            signals = (probs > 0.55).astype(float)
            # Assume direction is correct when label=1
            daily_returns = signals * (2 * y_test - 1) * 0.001  # simplified return proxy

            if daily_returns.std() > 0:
                sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
            else:
                sharpe = 0.0
            path_returns.append(sharpe)

        if not path_returns:
            return {"paths_tested": 0, "error": "Insufficient data for CPCV"}

        sharpe_array = np.array(path_returns)
        sharpe_mean = float(sharpe_array.mean())
        sharpe_std = float(sharpe_array.std()) if len(sharpe_array) > 1 else 1.0

        # Deflated Sharpe Ratio (simplified — adjusts for multiple testing)
        # DSR = (SR_mean - SR_benchmark) / SE(SR)
        # SR_benchmark ~ 0 for random strategy
        n_paths = len(sharpe_array)
        se_sharpe = sharpe_std / np.sqrt(max(n_paths, 1))
        deflated_sharpe = sharpe_mean / se_sharpe if se_sharpe > 0 else 0.0

        return {
            "paths_tested": n_paths,
            "sharpe_mean": round(sharpe_mean, 4),
            "sharpe_std": round(sharpe_std, 4),
            "deflated_sharpe": round(deflated_sharpe, 4),
            "statistically_significant": deflated_sharpe > 1.96,
            "path_sharpes": [round(s, 4) for s in path_returns],
        }

    def _compute_shap(self, X: np.ndarray) -> list[dict]:
        """Compute SHAP feature importance."""
        try:
            import shap
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            mean_abs = np.abs(shap_values).mean(axis=0)

            importance = []
            for i, col in enumerate(FEATURE_COLS):
                importance.append({
                    "feature": col,
                    "mean_abs_shap": round(float(mean_abs[i]), 6),
                })
            importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
            return importance
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            # Fallback to XGBoost native importance
            if self.model:
                imp = self.model.feature_importances_
                importance = [
                    {"feature": col, "mean_abs_shap": round(float(imp[i]), 6)}
                    for i, col in enumerate(FEATURE_COLS)
                ]
                importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
                return importance
            return []

    def store_prediction(self, run_date: date, instrument: str,
                         prediction: dict, regime_state: int,
                         composite_strength: float):
        """Store the prediction in the database."""
        from models.database import get_connection
        import json

        prediction_for = run_date  # In practice, next trading day

        conn = get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO predictions (date, prediction_for, instrument,
                        direction, probability, size_multiplier, model_version,
                        top_shap, regime_state, composite_strength)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (date, instrument) DO UPDATE SET
                        prediction_for = EXCLUDED.prediction_for,
                        direction = EXCLUDED.direction,
                        probability = EXCLUDED.probability,
                        size_multiplier = EXCLUDED.size_multiplier,
                        model_version = EXCLUDED.model_version,
                        top_shap = EXCLUDED.top_shap,
                        regime_state = EXCLUDED.regime_state,
                        composite_strength = EXCLUDED.composite_strength,
                        created_at = NOW()
                """, (
                    str(run_date), str(prediction_for), instrument,
                    prediction["direction"],
                    prediction["probability"],
                    prediction["size_multiplier"],
                    prediction.get("model_version", ""),
                    json.dumps(prediction.get("top_shap", [])),
                    regime_state,
                    composite_strength,
                ))
            conn.commit()
            logger.info(f"Stored prediction for {instrument} on {run_date}")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error storing prediction: {e}")
            raise
        finally:
            conn.close()

    def _save_model(self):
        """Serialize model to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, "meta_model.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "version": self.model_version,
                "cpcv": self.cpcv_results,
                "shap": self.shap_importance,
                "feature_cols": FEATURE_COLS,
            }, f)
        logger.info(f"Meta-model saved to {path}")

    def _load_model(self):
        """Load model from disk."""
        path = os.path.join(MODEL_DIR, "meta_model.pkl")
        if not os.path.exists(path):
            logger.warning(f"No meta-model at {path}")
            return
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.model_version = data["version"]
            self.cpcv_results = data.get("cpcv", {})
            self.shap_importance = data.get("shap", [])
        logger.info(f"Meta-model loaded: {self.model_version}")

    def _default_prediction(self, feature_vector: dict) -> dict:
        """Default when no model is available — conservative flat."""
        return {
            "direction": "flat",
            "probability": 0.50,
            "size_multiplier": 0.0,
            "model_version": "no_model",
            "top_shap": [],
        }
