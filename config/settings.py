"""
Quant EOD Engine — Configuration
All secrets loaded from environment variables or Docker secrets.
"""
import os
from pathlib import Path


def _read_secret(env_var: str, secret_file_var: str | None = None, default: str = "") -> str:
    """Read a secret from env var, Docker secret file, or default."""
    value = os.environ.get(env_var, "")
    if value:
        return value
    if secret_file_var:
        path = os.environ.get(secret_file_var, "")
        if path and Path(path).exists():
            return Path(path).read_text().strip()
    return default


# ─── OANDA ────────────────────────────────────────────────
OANDA_API_TOKEN = _read_secret("OANDA_API_TOKEN", "OANDA_TOKEN_FILE")
OANDA_ACCOUNT_ID = _read_secret("OANDA_ACCOUNT_ID", "OANDA_ACCOUNT_ID_FILE")
OANDA_BASE_URL = os.environ.get("OANDA_BASE_URL", "https://api-fxpractice.oanda.com")
# Practice: https://api-fxpractice.oanda.com
# Live:     https://api-fxtrade.oanda.com

# ─── FRED ─────────────────────────────────────────────────
FRED_API_KEY = _read_secret("FRED_API_KEY", "FRED_API_KEY_FILE")

# ─── Perplexity ───────────────────────────────────────────
PERPLEXITY_API_KEY = _read_secret("PERPLEXITY_API_KEY", "PERPLEXITY_API_KEY_FILE")
PERPLEXITY_MODEL = os.environ.get("PERPLEXITY_MODEL", "sonar-pro")
PERPLEXITY_BASE_URL = "https://api.perplexity.ai"

# ─── Discord ──────────────────────────────────────────────
DISCORD_WEBHOOK_URL = _read_secret("DISCORD_WEBHOOK_URL", "DISCORD_WEBHOOK_URL_FILE")

# ─── PostgreSQL ───────────────────────────────────────────
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5432"))
DB_NAME = os.environ.get("DB_NAME", "quant_eod")
DB_USER = os.environ.get("DB_USER", "postgres")
DB_PASSWORD = _read_secret("DB_PASSWORD", "DB_PASSWORD_FILE", "postgres")

DATABASE_URL = (
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# ─── Trading Instruments ──────────────────────────────────
INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY"]
PRIMARY_INSTRUMENT = "EUR_USD"

# ─── FRED Series IDs ─────────────────────────────────────
FRED_US_2Y_SERIES = "DGS2"          # US 2-Year Treasury yield
FRED_DE_2Y_SERIES = "DFII5"         # Proxy — will need adjustment (see note below)
# Note: FRED does not carry German 2Y directly. Options:
#   1. Use "IRLTLT01DEM156N" (long-term) as proxy
#   2. Use ECB Statistical Data Warehouse API
#   3. Use Trading Economics API
#   4. Hardcode a manual feed initially
# For MVP, we'll use what FRED has and document the gap.

# ─── Logging ──────────────────────────────────────────────
LOG_DIR = os.environ.get("LOG_DIR", "/home/user/workspace/quant-eod-engine/logs")
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
