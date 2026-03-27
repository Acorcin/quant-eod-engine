#!/usr/bin/env python3
"""
Quant EOD Engine — Daily Loop Orchestrator

This is the master script that runs the entire Phase 1 data collection pipeline.
Designed to be triggered by cron at 4:30 PM EST, Monday–Friday.

Cron entry (UTC — 4:30 PM EST = 20:30 UTC during EDT, 21:30 during EST):
  30 20 * * 1-5  cd /path/to/quant-eod-engine && python daily_loop.py

Pipeline steps:
  1. Pull OANDA bars (daily + 4H) for all instruments
  2. Pull FRED yield data
  3. Pull OANDA sentiment/position ratios
  4. Pull swap/financing rates
  5. Pull economic calendar
  6. Run Perplexity AI sentiment analysis
  7. Assemble and store daily snapshot
  8. Log pipeline run status
"""
import sys
import os
import json
import logging
from datetime import datetime, date, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import LOG_DIR, LOG_LEVEL, PRIMARY_INSTRUMENT
from models.database import init_schema, get_connection
from fetchers.oanda_bars import fetch_and_store_all as fetch_bars
from fetchers.fred_yields import fetch_and_store as fetch_yields
from fetchers.oanda_sentiment import fetch_and_store_all as fetch_sentiment
from fetchers.swap_rates import fetch_and_store_all as fetch_swaps
from fetchers.calendar import fetch_and_store as fetch_calendar
from fetchers.perplexity_sentiment import fetch_and_store as fetch_ai_sentiment
from fetchers.discord_notify import send_signal, send_error_alert

# ─── Logging Setup ────────────────────────────────────────
os.makedirs(LOG_DIR, exist_ok=True)
log_file = os.path.join(LOG_DIR, f"daily_{date.today().isoformat()}.log")

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("daily_loop")


def log_pipeline_run(run_date: date, started_at: datetime, status: str,
                     steps: dict, errors: dict):
    """Record the pipeline run in the database."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO pipeline_runs (run_date, started_at, completed_at, status, steps_completed, errors)
                VALUES (%s, %s, NOW(), %s, %s, %s)
                ON CONFLICT (run_date) DO UPDATE SET
                    completed_at = NOW(),
                    status = EXCLUDED.status,
                    steps_completed = EXCLUDED.steps_completed,
                    errors = EXCLUDED.errors
            """, (
                str(run_date), started_at, status,
                json.dumps(steps), json.dumps(errors),
            ))
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to log pipeline run: {e}")
    finally:
        conn.close()


def assemble_daily_snapshot(bars_result, yields_result, sentiment_result,
                            swaps_result, calendar_result, ai_result) -> dict:
    """Assemble all collected data into a single daily snapshot JSON."""
    today = date.today()

    snapshot = {
        "type": "daily_snapshot",
        "date": str(today),
        "collection_time": datetime.now(timezone.utc).isoformat(),
        "bars_summary": bars_result,
        "macro": yields_result if isinstance(yields_result, dict) else {},
        "sentiment": sentiment_result if isinstance(sentiment_result, dict) else {},
        "swap_rates": swaps_result if isinstance(swaps_result, dict) else {},
        "calendar": calendar_result if isinstance(calendar_result, dict) else {},
        "ai_sentiment": {
            k: v for k, v in (ai_result or {}).items()
            if k != "raw_response"  # don't duplicate the full response in snapshot
        },
        "is_friday": today.weekday() == 4,
    }

    return snapshot


def store_snapshot(snapshot: dict):
    """Store the assembled snapshot in the database."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO daily_snapshots (date, instrument, snapshot_data)
                VALUES (%s, %s, %s)
                ON CONFLICT (date, instrument) DO UPDATE SET
                    snapshot_data = EXCLUDED.snapshot_data,
                    created_at = NOW()
            """, (
                snapshot["date"], PRIMARY_INSTRUMENT,
                json.dumps(snapshot, default=str),
            ))
        conn.commit()
        logger.info(f"Stored daily snapshot for {snapshot['date']}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing snapshot: {e}")
    finally:
        conn.close()


def main():
    """Run the full daily data collection pipeline."""
    started_at = datetime.now(timezone.utc)
    today = date.today()
    steps = {}
    errors = {}

    logger.info("=" * 60)
    logger.info(f"DAILY LOOP STARTED — {today} ({today.strftime('%A')})")
    logger.info("=" * 60)

    # Step 0: Ensure DB schema exists
    try:
        init_schema()
        steps["schema_init"] = True
    except Exception as e:
        logger.error(f"Schema init failed: {e}")
        errors["schema_init"] = str(e)
        # Continue — schema might already exist

    # Step 1: OANDA Bars
    logger.info("─── Step 1: Fetching OANDA bars ───")
    try:
        bars_result = fetch_bars()
        steps["oanda_bars"] = True
        logger.info(f"Bars result: {bars_result}")
    except Exception as e:
        logger.error(f"OANDA bars failed: {e}")
        bars_result = {"error": str(e)}
        errors["oanda_bars"] = str(e)

    # Step 2: FRED Yields
    logger.info("─── Step 2: Fetching FRED yields ───")
    try:
        yields_result = fetch_yields()
        steps["fred_yields"] = True
        logger.info(f"Yields result: {yields_result}")
    except Exception as e:
        logger.error(f"FRED yields failed: {e}")
        yields_result = {"error": str(e)}
        errors["fred_yields"] = str(e)

    # Step 3: Sentiment
    logger.info("─── Step 3: Fetching sentiment ───")
    try:
        sentiment_result = fetch_sentiment()
        steps["sentiment"] = True
    except Exception as e:
        logger.error(f"Sentiment failed: {e}")
        sentiment_result = {"error": str(e)}
        errors["sentiment"] = str(e)

    # Step 4: Swap Rates
    logger.info("─── Step 4: Fetching swap rates ───")
    try:
        swaps_result = fetch_swaps()
        steps["swap_rates"] = True
    except Exception as e:
        logger.error(f"Swap rates failed: {e}")
        swaps_result = {"error": str(e)}
        errors["swap_rates"] = str(e)

    # Step 5: Economic Calendar
    logger.info("─── Step 5: Fetching calendar ───")
    try:
        calendar_result = fetch_calendar()
        steps["calendar"] = True
    except Exception as e:
        logger.error(f"Calendar failed: {e}")
        calendar_result = {"error": str(e)}
        errors["calendar"] = str(e)

    # Step 6: Perplexity AI Sentiment
    logger.info("─── Step 6: Fetching AI sentiment (Perplexity) ───")
    try:
        ai_result = fetch_ai_sentiment()
        steps["ai_sentiment"] = True
        logger.info(
            f"AI sentiment: score={ai_result.get('macro_sentiment_score')}, "
            f"confidence={ai_result.get('confidence')}"
        )
    except Exception as e:
        logger.error(f"AI sentiment failed: {e}")
        ai_result = {"error": str(e)}
        errors["ai_sentiment"] = str(e)

    # Step 7: Assemble Snapshot
    logger.info("─── Step 7: Assembling daily snapshot ───")
    try:
        snapshot = assemble_daily_snapshot(
            bars_result, yields_result, sentiment_result,
            swaps_result, calendar_result, ai_result,
        )
        store_snapshot(snapshot)
        steps["snapshot"] = True
    except Exception as e:
        logger.error(f"Snapshot assembly failed: {e}")
        errors["snapshot"] = str(e)

    # Determine overall status
    if not errors:
        status = "success"
    elif len(errors) < 3:
        status = "partial"
    else:
        status = "failed"

    # Log the run
    log_pipeline_run(today, started_at, status, steps, errors)

    # Step 8: Send Discord notification
    logger.info("─── Step 8: Sending Discord notification ───")
    try:
        if 'snapshot' in locals() and snapshot:
            sent = send_signal(snapshot, status)
        else:
            sent = send_error_alert(f"Pipeline {status}: {json.dumps(errors)}")
        if sent:
            steps["discord_notify"] = True
        else:
            logger.warning("Discord notification skipped (no webhook configured or send failed)")
    except Exception as e:
        logger.error(f"Discord notification failed: {e}")
        errors["discord_notify"] = str(e)

    logger.info("=" * 60)
    logger.info(f"DAILY LOOP COMPLETED — Status: {status.upper()}")
    logger.info(f"Steps OK: {list(steps.keys())}")
    if errors:
        logger.warning(f"Errors: {list(errors.keys())}")
    logger.info("=" * 60)

    return status


if __name__ == "__main__":
    status = main()
    sys.exit(0 if status in ("success", "partial") else 1)
