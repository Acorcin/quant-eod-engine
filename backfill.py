#!/usr/bin/env python3
"""
Historical Backfill Script.

Fetches 2 years of daily OANDA candle data and stores it in the
database. This bootstraps the HMM regime detector and provides
enough history for the meta-model to train on.

Usage:
    python backfill.py [--days 504]
"""
import sys
import os
import argparse
import logging
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import OANDA_API_TOKEN, OANDA_BASE_URL, INSTRUMENTS
from fetchers.oanda_bars import fetch_candles, store_candles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("backfill")


def backfill_bars(days: int = 504):
    """
    Backfill daily bars for all instruments.

    OANDA allows max 5000 bars per request, so 504 days (2 years
    of trading days) is well within limits.

    Args:
        days: Number of daily bars to fetch (default 504 = ~2 years).
    """
    logger.info(f"Starting backfill: {days} daily bars for {INSTRUMENTS}")

    for instrument in INSTRUMENTS:
        try:
            logger.info(f"Fetching {days} daily bars for {instrument}...")
            candles = fetch_candles(instrument, "D", days)
            store_candles(candles)
            logger.info(f"  Stored {len(candles)} daily bars for {instrument}")

            # Also fetch extended 4H history (6 bars/day × days)
            h4_count = min(days * 6, 5000)
            logger.info(f"Fetching {h4_count} 4H bars for {instrument}...")
            h4_candles = fetch_candles(instrument, "H4", h4_count)
            store_candles(h4_candles)
            logger.info(f"  Stored {len(h4_candles)} 4H bars for {instrument}")

        except Exception as e:
            logger.error(f"Backfill failed for {instrument}: {e}")

    logger.info("Backfill complete.")


def backfill_and_fit_hmm():
    """Backfill bars then fit the HMM on the history."""
    from models.database import init_schema
    from models.hmm_regime import RegimeDetector

    # Ensure Phase 2 schema exists
    schema_path = os.path.join(os.path.dirname(__file__), "sql", "schema_phase2.sql")
    if os.path.exists(schema_path):
        from models.database import get_connection
        conn = get_connection()
        try:
            with open(schema_path) as f:
                sql = f.read()
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()
            logger.info("Phase 2 schema initialized")
        except Exception as e:
            conn.rollback()
            logger.warning(f"Phase 2 schema init: {e}")
        finally:
            conn.close()

    # Backfill bars
    backfill_bars()

    # Fit HMM
    logger.info("Fitting HMM regime detector on backfilled data...")
    detector = RegimeDetector()
    version = detector.fit("EUR_USD")
    regime = detector.predict_regime("EUR_USD")
    logger.info(f"HMM fitted: version={version}")
    logger.info(f"Current regime: {regime['state_label']} (conf={regime['confidence']:.3f})")

    return regime


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical data")
    parser.add_argument("--days", type=int, default=504, help="Days of history (default 504)")
    parser.add_argument("--hmm", action="store_true", help="Also fit HMM after backfill")
    args = parser.parse_args()

    if args.hmm:
        backfill_and_fit_hmm()
    else:
        backfill_bars(args.days)
