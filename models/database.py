"""
Database connection and helper utilities.
"""
import psycopg2
import psycopg2.extras
import logging
from config.settings import DATABASE_URL

logger = logging.getLogger(__name__)


def get_connection():
    """Get a new database connection."""
    return psycopg2.connect(DATABASE_URL)


def execute(query: str, params: tuple = None):
    """Execute a query (INSERT/UPDATE/DELETE)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"DB execute error: {e}")
        raise
    finally:
        conn.close()


def execute_returning(query: str, params: tuple = None):
    """Execute a query and return the result."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            result = cur.fetchone()
        conn.commit()
        return result
    except Exception as e:
        conn.rollback()
        logger.error(f"DB execute_returning error: {e}")
        raise
    finally:
        conn.close()


def fetch_all(query: str, params: tuple = None) -> list[dict]:
    """Fetch all rows as list of dicts."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]
    finally:
        conn.close()


def fetch_one(query: str, params: tuple = None) -> dict | None:
    """Fetch a single row as dict."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, params)
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def init_schema():
    """Initialize the database schema from sql/schema.sql."""
    import os
    schema_path = os.path.join(os.path.dirname(__file__), "..", "sql", "schema.sql")
    with open(schema_path) as f:
        sql = f.read()
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
        logger.info("Database schema initialized successfully")
    except Exception as e:
        conn.rollback()
        logger.error(f"Schema init error: {e}")
        raise
    finally:
        conn.close()
