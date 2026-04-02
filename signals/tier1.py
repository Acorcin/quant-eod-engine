"""
Tier 1 Signal Generators — Primary Trading Signals.

These are the core alpha signals. Each generates an independent
directional signal (long/short/flat) with a strength score.

1. Yield Spread Momentum — macro rate differential dynamics
2. Sentiment Extreme Fade — contrarian fade of retail crowd
3. AI Macro Sentiment — Perplexity LLM-scored macro analysis
4. EOD Event Reversal — institutional pattern after high-impact events
"""
import logging
from datetime import date
from models.database import fetch_one, fetch_all
from config.settings import SENTIMENT_EXTREME_HIGH, SENTIMENT_EXTREME_LOW

logger = logging.getLogger(__name__)

# Regime-adaptive thresholds
YIELD_THRESHOLDS = {
    0: 8.0,   # low_vol_trend: tighter threshold (smaller moves matter)
    1: 15.0,  # high_vol_choppy: wider (need bigger moves to be meaningful)
    2: 20.0,  # high_vol_crash: widest (extreme noise)
}


def yield_spread_momentum(run_date: date, instrument: str, regime_state: int) -> dict:
    """
    Yield Spread Momentum Signal.

    Logic:
    - If 5d yield spread change > threshold AND favors USD → SHORT EUR/USD
    - If 5d yield spread change < -threshold AND favors EUR → LONG EUR/USD
    - Threshold is regime-adaptive (tighter in trends, wider in chop)
    """
    macro = fetch_one(
        "SELECT * FROM yield_data WHERE date <= %s ORDER BY date DESC LIMIT 1",
        (str(run_date),),
    )
    if not macro:
        return _no_signal("yield_spread_momentum", "No macro data available")

    raw_spread = macro.get("spread_change_5d_bps")
    spread_change_5d = float(
        raw_spread if raw_spread is not None else (macro.get("us_2y_change_5d_bps") or 0)
    )
    threshold = YIELD_THRESHOLDS.get(regime_state, 15.0)

    if spread_change_5d > threshold:
        # US yields rising relative → USD strength → SHORT EUR/USD
        strength = min(abs(spread_change_5d) / (threshold * 2), 1.0)
        return _signal(
            "yield_spread_momentum", "short", strength,
            f"5d spread change +{spread_change_5d:.1f} bps > threshold {threshold} → USD strength",
            {"spread_change_5d_bps": spread_change_5d, "threshold": threshold},
        )
    elif spread_change_5d < -threshold:
        # US yields falling relative → EUR strength → LONG EUR/USD
        strength = min(abs(spread_change_5d) / (threshold * 2), 1.0)
        return _signal(
            "yield_spread_momentum", "long", strength,
            f"5d spread change {spread_change_5d:.1f} bps < -{threshold} → EUR strength",
            {"spread_change_5d_bps": spread_change_5d, "threshold": threshold},
        )
    else:
        return _signal(
            "yield_spread_momentum", "flat", 0.0,
            f"5d spread change {spread_change_5d:.1f} bps within ±{threshold} threshold",
            {"spread_change_5d_bps": spread_change_5d, "threshold": threshold},
        )


def sentiment_extreme_fade(run_date: date, instrument: str) -> dict:
    """
    Retail Sentiment Extreme — Fade the Crowd.

    Logic:
    - If retail > high threshold → SHORT (fade the longs)
    - If retail < low threshold → LONG (fade the shorts)
    - Retail traders are demonstrably wrong at extremes
    """
    row = fetch_one(
        "SELECT * FROM sentiment WHERE instrument = %s AND date <= %s ORDER BY date DESC LIMIT 1",
        (instrument, str(run_date)),
    )
    if not row:
        return _no_signal("sentiment_extreme_fade", "No sentiment data")

    pct_long = float(row.get("pct_long") or 0.5)

    high = SENTIMENT_EXTREME_HIGH
    low = SENTIMENT_EXTREME_LOW
    high_span = max(1.0 - high, 1e-6)
    low_span = max(low, 1e-6)

    if pct_long > high:
        strength = min((pct_long - high) / high_span, 1.0)
        return _signal(
            "sentiment_extreme_fade", "short", strength,
            f"Retail {pct_long*100:.1f}% long — extreme, fade longs",
            {"pct_long": pct_long, "threshold_high": high},
        )
    elif pct_long < low:
        strength = min((low - pct_long) / low_span, 1.0)
        return _signal(
            "sentiment_extreme_fade", "long", strength,
            f"Retail {pct_long*100:.1f}% long — extreme short positioning, fade shorts",
            {"pct_long": pct_long, "threshold_low": low},
        )
    else:
        return _signal(
            "sentiment_extreme_fade", "flat", 0.0,
            f"Retail {pct_long*100:.1f}% long — within normal range",
            {"pct_long": pct_long},
        )


def ai_macro_sentiment(run_date: date) -> dict:
    """
    AI Macro Sentiment Signal.

    Logic:
    - If Perplexity score < -0.5 AND confidence > 0.6 → SHORT EUR/USD
    - If Perplexity score > 0.5 AND confidence > 0.6 → LONG EUR/USD
    - This is the unique edge: no other retail quant reads the internet daily via LLM
    """
    row = fetch_one(
        "SELECT * FROM ai_sentiment WHERE date = %s",
        (str(run_date),),
    )
    if not row:
        return _no_signal("ai_macro_sentiment", "No AI sentiment for today")

    score = float(row.get("macro_sentiment_score") or 0)
    confidence = float(row.get("confidence") or 0)
    driver = row.get("dominant_driver", "unknown")
    fallback = row.get("fallback_used", True)

    if fallback:
        return _signal(
            "ai_macro_sentiment", "flat", 0.0,
            "AI sentiment used fallback — no signal generated",
            {"score": score, "confidence": confidence, "fallback": True},
        )

    if score < -0.5 and confidence > 0.6:
        strength = min(abs(score) * confidence, 1.0)
        return _signal(
            "ai_macro_sentiment", "short", strength,
            f"AI score {score:.2f} (conf {confidence:.2f}), driver: {driver} → bearish EUR/USD",
            {"score": score, "confidence": confidence, "driver": driver},
        )
    elif score > 0.5 and confidence > 0.6:
        strength = min(abs(score) * confidence, 1.0)
        return _signal(
            "ai_macro_sentiment", "long", strength,
            f"AI score {score:.2f} (conf {confidence:.2f}), driver: {driver} → bullish EUR/USD",
            {"score": score, "confidence": confidence, "driver": driver},
        )
    else:
        return _signal(
            "ai_macro_sentiment", "flat", 0.0,
            f"AI score {score:.2f} (conf {confidence:.2f}) — below threshold or low confidence",
            {"score": score, "confidence": confidence, "driver": driver},
        )


def eod_event_reversal(run_date: date, instrument: str, technical: dict) -> dict:
    """
    EOD Event Reversal Signal.

    Logic:
    - Was there a high-impact event today?
    - Did the event surprise in one direction (e.g., bullish USD)?
    - Did the daily candle CLOSE in the OPPOSITE direction?
    - If yes → institutional signal: market absorbed the news and reversed.

    This is a well-documented institutional pattern:
    "The market tells you what it thinks of the news by the close."
    """
    events = fetch_all(
        """SELECT * FROM calendar_events
           WHERE DATE(event_time) = %s AND impact = 'high'
           ORDER BY event_time""",
        (str(run_date),),
    )

    if not events:
        return _signal(
            "eod_event_reversal", "flat", 0.0,
            "No high-impact events today",
            {"triggered": False},
        )

    def _usd_surprise_score(sd: str) -> float:
        if not sd or sd == "neutral":
            return 0.0
        if "positive_usd" in sd:
            return 1.0
        if "negative_usd" in sd:
            return -1.0
        return 0.0

    surprise_scores = [_usd_surprise_score(str(e.get("surprise_direction") or "")) for e in events]
    net_usd = float(sum(surprise_scores))
    non_neutral = [e for e in events if (e.get("surprise_direction") or "") not in ("", "neutral")]

    if not non_neutral:
        return _signal(
            "eod_event_reversal", "flat", 0.0,
            "Events occurred but no surprise_direction set",
            {"triggered": False, "events_count": len(events)},
        )

    if net_usd == 0.0:
        return _signal(
            "eod_event_reversal", "flat", 0.0,
            "High-impact events conflict (net USD surprise ≈ 0)",
            {
                "triggered": False,
                "events_count": len(events),
                "conflicting_surprises": True,
                "per_event_scores": surprise_scores,
            },
        )

    surprise_direction = "positive_usd" if net_usd > 0 else "negative_usd"

    body_dir = technical.get("body_direction", 0)

    # Determine if there's a reversal
    # positive_usd surprise + bullish EUR candle = reversal → LONG EUR/USD
    # negative_usd surprise + bearish EUR candle = reversal → SHORT EUR/USD
    usd_positive = "positive_usd" in surprise_direction
    usd_negative = "negative_usd" in surprise_direction

    reversal_detected = False
    direction = "flat"

    if usd_positive and body_dir == 1:  # USD-positive event, but EUR candle is bullish
        reversal_detected = True
        direction = "long"
    elif usd_negative and body_dir == -1:  # USD-negative event, but EUR candle is bearish
        reversal_detected = True
        direction = "short"

    if reversal_detected:
        strength = min(0.85 + 0.02 * (len(non_neutral) - 1), 1.0)
        return _signal(
            "eod_event_reversal", direction, strength,
            f"Aggregated USD surprise ({surprise_direction}) vs candle → institutional reversal",
            {
                "triggered": True,
                "surprise": surprise_direction,
                "net_usd_score": net_usd,
                "events_count": len(events),
                "candle_direction": body_dir,
            },
        )
    else:
        return _signal(
            "eod_event_reversal", "flat", 0.0,
            f"Aggregated surprise '{surprise_direction}' aligned with candle — no reversal",
            {
                "triggered": False,
                "surprise": surprise_direction,
                "net_usd_score": net_usd,
                "candle_direction": body_dir,
            },
        )


def generate_all_tier1(run_date: date, instrument: str, regime_state: int, technical: dict) -> list[dict]:
    """Run all Tier 1 generators and return list of signals."""
    signals = [
        yield_spread_momentum(run_date, instrument, regime_state),
        sentiment_extreme_fade(run_date, instrument),
        ai_macro_sentiment(run_date),
        eod_event_reversal(run_date, instrument, technical),
    ]
    logger.info(f"Tier 1 signals: {[(s['detector'], s['direction'], s['strength']) for s in signals]}")
    return signals


# ─── Helpers ─────────────────────────────────────────────

def _signal(detector: str, direction: str, strength: float, detail: str, metadata: dict = None) -> dict:
    return {
        "tier": 1,
        "detector": detector,
        "direction": direction,
        "strength": round(strength, 4),
        "detail": detail,
        "metadata": metadata or {},
    }


def _no_signal(detector: str, reason: str) -> dict:
    return _signal(detector, "flat", 0.0, reason, {"no_data": True})
