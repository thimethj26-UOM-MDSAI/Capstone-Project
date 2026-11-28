import sqlite3, threading
from contextlib import closing
from pathlib import Path

DB_PATH = Path("moderation/moderation.db")
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS items(
 id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, parent_id INTEGER,
 author TEXT NOT NULL, text TEXT NOT NULL, created_at REAL NOT NULL, status TEXT NOT NULL,
 prob REAL, words TEXT, t_flag REAL, t_remove REAL, model TEXT, scanned_at REAL,
 reviewed INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS alerts(
 id INTEGER PRIMARY KEY AUTOINCREMENT, user TEXT NOT NULL, item_id INTEGER,
 message TEXT NOT NULL, created_at REAL NOT NULL, read INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS events(
 id INTEGER PRIMARY KEY AUTOINCREMENT, item_id INTEGER NOT NULL, event TEXT NOT NULL,
 detail TEXT, ts REAL NOT NULL);
"""


def _connect():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init():
    DB_PATH.parent.mkdir(exist_ok=True)
    with _lock, closing(_connect()) as c:
        c.executescript(SCHEMA)
        c.commit()


def q(sql, args=()):
    with _lock, closing(_connect()) as c:
        return [dict(r) for r in c.execute(sql, args).fetchall()]


def x(sql, args=()):
    with _lock, closing(_connect()) as c:
        cur = c.execute(sql, args)
        c.commit()
        return cur.lastrowid
