"""
Feature Vector Assembly.

Combines technical indicators, macro data, sentiment, AI analysis,
regime state, and signal outputs into the 26-feature vector that
feeds the XGBoost meta-model.
"""
import json
import logging
from datetime import date
from models.database import get_connection, fetch_one

logger = logging.getLogger(__name__)

# Encoding maps for categorical features
STANCE_MAP = {"hawkish": 1, "neutral": 0, "dovish": -1}
RISK_MAP = {"risk_on": 1, "neutral": 0, "risk_off": -1}


def assemble_feature_vector(
    run_date: date,
    instrument: str,
    technical: dict,
    regime: dict,
    signals_summary: dict,
) -> dict:
    """
    Assemble the full feature vector for the meta-model.

    Args:
        run_date: Today's date.
        instrument: e.g. "EUR_USD".
        technical: Output from features.technical.compute_all_features().
        regime: Regime dict with state_id, state_label, confidence, days_in_regime.
        signals_summary: Summary of signal outputs (direction, count, strength, etc.).

    Returns:
        Dict of feature name → value (the 26-feature vector).
    """
    # Pull macro data from DB
    macro = _get_macro_data(run_date)

    # Pull AI sentiment from DB
    ai = _get_ai_sentiment(run_date)

    # Pull sentiment from DB
    sentiment = _get_sentiment(run_date, instrument)

    # Pull swap rates from DB
    swaps = _get_swap_rates(run_date, instrument)

    vector = {
        # ─── Regime (from HMM) ───
        "regime_state": regime.get("state_id", 1),
        "days_in_regime": regime.get("days_in_regime", 1),

        # ─── Macro (from FRED) ───
        "yield_spread_bps": macro.get("yield_spread_bps", 0.0),
        "yield_spread_change_5d": macro.get("us_2y_change_5d_bps", 0.0),
        "yield_spread_change_20d": macro.get("us_2y_change_20d_bps", 0.0),

        # ─── Sentiment (from OANDA) ───
        "sentiment_pct_long": sentiment.get("pct_long", 0.5),
        "sentiment_extreme": 1 if _is_sentiment_extreme(sentiment.get("pct_long", 0.5)) else 0,

        # ─── AI Sentiment (from Perplexity) ───
        "macro_sentiment_score": ai.get("macro_sentiment_score", 0.0),
        "ai_confidence": ai.get("confidence", 0.1),
        "fed_stance_encoded": STANCE_MAP.get(ai.get("fed_stance", "neutral"), 0),
        "ecb_stance_encoded": STANCE_MAP.get(ai.get("ecb_stance", "neutral"), 0),
        "risk_sentiment_encoded": RISK_MAP.get(ai.get("risk_sentiment", "neutral"), 0),

        # ─── Technical (from bars) ───
        "atr_14": technical.get("atr_14", 0.0),
        "rsi_14": technical.get("rsi_14", 50.0),
        "price_vs_ma50": technical.get("price_vs_ma50", 0.0),
        "price_vs_ma200": technical.get("price_vs_ma200", 0.0),
        "body_direction": technical.get("body_direction", 0),
        "body_pct_of_range": technical.get("body_pct_of_range", 0.5),

        # ─── EOD Event Reversal ───
        "eod_event_reversal": signals_summary.get("eod_event_reversal", 0),
        "event_surprise_magnitude": signals_summary.get("event_surprise_magnitude", 0.0),

        # ─── Time Features ───
        "day_of_week": run_date.weekday(),  # 0=Mon, 4=Fri
        "is_friday": 1 if run_date.weekday() == 4 else 0,

        # ─── Cost Features ───
        "long_swap_pips": swaps.get("long_swap_pips", 0.0),
        "short_swap_pips": swaps.get("short_swap_pips", 0.0),

        # ─── Signal Summary ───
        "primary_signal_direction": signals_summary.get("direction_encoded", 0),
        "primary_signal_count": signals_summary.get("signal_count", 0),
        "composite_strength": signals_summary.get("composite_strength", 0.0),
        "tier2_confirmation_count": signals_summary.get("tier2_count", 0),
    }

    # Coerce None to 0.0 for model compatibility
    for k, v in vector.items():
        if v is None:
            vector[k] = 0.0

    return vector


def store_feature_vector(run_date: date, instrument: str, vector: dict):
    """Store the feature vector in the database."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO feature_vectors (date, instrument, features)
                VALUES (%s, %s, %s)
                ON CONFLICT (date, instrument) DO UPDATE SET
                    features = EXCLUDED.features,
                    created_at = NOW()
            """, (str(run_date), instrument, json.dumps(vector)))
        conn.commit()
        logger.info(f"Stored feature vector for {instrument} on {run_date}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing feature vector: {e}")
        raise
    finally:
        conn.close()


def _is_sentiment_extreme(pct_long: float) -> bool:
    """Check if sentiment is at an extreme (>72% or <28% long)."""
    return pct_long > 0.72 or pct_long < 0.28


def _get_macro_data(run_date: date) -> dict:
    """Pull latest macro/yield data from DB."""
    row = fetch_one(
        "SELECT * FROM yield_data WHERE date <= %s ORDER BY date DESC LIMIT 1",
        (str(run_date),),
    )
    return dict(row) if row else {}


def _get_ai_sentiment(run_date: date) -> dict:
    """Pull today's AI sentiment from DB."""
    row = fetch_one(
        "SELECT * FROM ai_sentiment WHERE date = %s",
        (str(run_date),),
    )
    return dict(row) if row else {}


def _get_sentiment(run_date: date, instrument: str) -> dict:
    """Pull latest retail sentiment from DB."""
    row = fetch_one(
        "SELECT * FROM sentiment WHERE instrument = %s AND date <= %s ORDER BY date DESC LIMIT 1",
        (instrument, str(run_date)),
    )
    return dict(row) if row else {}


def _get_swap_rates(run_date: date, instrument: str) -> dict:
    """Pull latest swap rates from DB."""
    row = fetch_one(
        "SELECT * FROM swap_rates WHERE instrument = %s AND date <= %s ORDER BY date DESC LIMIT 1",
        (instrument, str(run_date)),
    )
    return dict(row) if row else {}
