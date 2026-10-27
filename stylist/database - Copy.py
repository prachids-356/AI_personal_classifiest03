"""Tiny SQLite store for saved outfits and like/dislike feedback."""
from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "stylist.db"


def _conn():
    return sqlite3.connect(DB_PATH)


def create_tables():
    with closing(_conn()) as c, c:
        c.execute("""CREATE TABLE IF NOT EXISTS saved_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_name TEXT, item_id INTEGER,
            name TEXT, image_path TEXT, saved_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_name, item_id))""")
        c.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT, user_name TEXT, item_id INTEGER,
            feedback TEXT, UNIQUE(user_name, item_id))""")


def save_item(user, item_id, name, image_path):
    with closing(_conn()) as c, c:
        c.execute("INSERT OR IGNORE INTO saved_items (user_name,item_id,name,image_path) VALUES (?,?,?,?)",
                  (user, int(item_id), str(name), str(image_path)))


def get_saved(user):
    with closing(_conn()) as c:
        rows = c.execute("SELECT item_id,name,image_path FROM saved_items WHERE user_name=? ORDER BY id DESC",
                         (user,)).fetchall()
    return [{"item_id": r[0], "name": r[1], "image_path": r[2]} for r in rows]


def save_feedback(user, item_id, feedback):
    """feedback: 'Like' or 'Dislike' (replaces any earlier vote on that item)."""
    with closing(_conn()) as c, c:
        c.execute("INSERT OR REPLACE INTO feedback (user_name,item_id,feedback) VALUES (?,?,?)",
                  (user, int(item_id), feedback))


def get_feedback_scores(user):
    with closing(_conn()) as c:
        rows = c.execute("SELECT item_id,feedback FROM feedback WHERE user_name=?", (user,)).fetchall()
    return {i: (1 if f == "Like" else -1) for i, f in rows}
