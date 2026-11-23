import os
import sqlite3

from datetime import datetime


DB_PATH = "/app/runs/tilletia.sqlite3"


def connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def has_column(connection, table_name, column_name):
    rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(row["name"] == column_name for row in rows)


def utcnow():
    return datetime.utcnow()


def utcnow_text():
    return utcnow().isoformat() + "Z"
