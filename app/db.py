from contextlib import contextmanager

import psycopg2
import psycopg2.extras

from app.config import settings


def get_connection():
    if not settings.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg2.connect(settings.DATABASE_URL)


@contextmanager
def db_cursor(commit: bool = False):
    conn = get_connection()
    try:
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        yield cursor
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()