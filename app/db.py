from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row

from app.config import settings


def get_connection():
    if not settings.DATABASE_URL:
        raise RuntimeError("DATABASE_URL is not set")
    return psycopg.connect(settings.DATABASE_URL, row_factory=dict_row)


@contextmanager
def db_cursor(commit: bool = False):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            yield cursor
        if commit:
            conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()