"""
Fetcher: FRED API — US and German 2-Year bond yields.

Computes the yield spread (US 2Y - DE 2Y) and rate-of-change metrics.
Uses the fredapi library for clean data access.
"""
import logging
from datetime import datetime, timedelta, date
from fredapi import Fred
from config.settings import FRED_API_KEY, FRED_US_2Y_SERIES, FRED_DE_2Y_SERIES
from models.database import get_connection

logger = logging.getLogger(__name__)


def fetch_yields(lookback_days: int = 30) -> dict:
    """
    Fetch US 2Y and German 2Y yields from FRED.

    Returns dict with latest yields and spread calculations.
    """
    fred = Fred(api_key=FRED_API_KEY)
    end = date.today()
    start = end - timedelta(days=lookback_days + 10)  # extra buffer for weekends/holidays

    # Fetch US 2-Year Treasury
    try:
        us_2y = fred.get_series(FRED_US_2Y_SERIES, observation_start=start, observation_end=end)
        us_2y = us_2y.dropna()
        logger.info(f"Fetched {len(us_2y)} US 2Y observations")
    except Exception as e:
        logger.error(f"Failed to fetch US 2Y yield: {e}")
        us_2y = None

    # Fetch German 2Y proxy
    # Note: FRED doesn't carry German 2Y directly.
    # Using FRED series or fallback value — see settings.py for alternatives.
    de_2y = None
    try:
        de_2y = fred.get_series(FRED_DE_2Y_SERIES, observation_start=start, observation_end=end)
        de_2y = de_2y.dropna()
        logger.info(f"Fetched {len(de_2y)} DE 2Y observations")
    except Exception as e:
        logger.warning(f"German 2Y series unavailable from FRED: {e}. Using fallback.")

    if us_2y is None or us_2y.empty:
        return {"error": "US 2Y yield data unavailable"}

    # Get latest values
    latest_us = float(us_2y.iloc[-1])
    latest_date = us_2y.index[-1].date()

    # German yield: use fetched data or document the gap
    latest_de = None
    if de_2y is not None and not de_2y.empty:
        latest_de = float(de_2y.iloc[-1])
    else:
        # Fallback: you'll need to manually set this or add an alternative source
        latest_de = None
        logger.warning("German 2Y yield unavailable — spread calculations will be partial")

    # Compute spread
    spread_bps = None
    if latest_de is not None:
        spread_bps = round((latest_us - latest_de) * 100, 2)  # convert to basis points

    # Compute rate of change (using US 2Y as primary indicator)
    change_1d = None
    change_5d = None
    change_20d = None

    if len(us_2y) >= 2:
        change_1d = round((latest_us - float(us_2y.iloc[-2])) * 100, 2)
    if len(us_2y) >= 6:
        change_5d = round((latest_us - float(us_2y.iloc[-6])) * 100, 2)
    if len(us_2y) >= 21:
        change_20d = round((latest_us - float(us_2y.iloc[-21])) * 100, 2)

    result = {
        "date": str(latest_date),
        "us_2y_yield": latest_us,
        "de_2y_yield": latest_de,
        "yield_spread_bps": spread_bps,
        "us_2y_change_1d_bps": change_1d,
        "us_2y_change_5d_bps": change_5d,
        "us_2y_change_20d_bps": change_20d,
        "source": "fred",
    }

    logger.info(f"Yield data: US 2Y={latest_us}, spread={spread_bps} bps")
    return result


def store_yields(data: dict):
    """Store yield data in the database."""
    if "error" in data:
        logger.warning(f"Skipping yield storage: {data['error']}")
        return

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO yield_data (date, us_2y_yield, de_2y_yield, yield_spread_bps, source)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (date, source)
                DO UPDATE SET
                    us_2y_yield = EXCLUDED.us_2y_yield,
                    de_2y_yield = EXCLUDED.de_2y_yield,
                    yield_spread_bps = EXCLUDED.yield_spread_bps,
                    fetched_at = NOW()
            """, (
                data["date"], data["us_2y_yield"], data["de_2y_yield"],
                data["yield_spread_bps"], data["source"],
            ))
        conn.commit()
        logger.info(f"Stored yield data for {data['date']}")
    except Exception as e:
        conn.rollback()
        logger.error(f"Error storing yields: {e}")
        raise
    finally:
        conn.close()


def fetch_and_store():
    """Fetch yields from FRED and store in DB."""
    data = fetch_yields()
    store_yields(data)
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = fetch_and_store()
    print(result)
