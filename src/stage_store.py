import json

import db


def init_schema(connection):
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS stage_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            speed_mm_s REAL NOT NULL DEFAULT 5,
            step_mm REAL NOT NULL DEFAULT 0.5,
            min_x REAL NOT NULL DEFAULT -2,
            max_x REAL NOT NULL DEFAULT 8,
            min_y REAL NOT NULL DEFAULT -2,
            max_y REAL NOT NULL DEFAULT 8,
            machine_x REAL NOT NULL DEFAULT 0,
            machine_y REAL NOT NULL DEFAULT 0,
            zero_x REAL NOT NULL DEFAULT 0,
            zero_y REAL NOT NULL DEFAULT 0,
            zero_set INTEGER NOT NULL DEFAULT 0,
            points_json TEXT NOT NULL DEFAULT '[]',
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS stage_routes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            speed_mm_s REAL NOT NULL,
            points_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """
    )
    if connection.execute("SELECT id FROM stage_state WHERE id = 1").fetchone() is None:
        connection.execute(
            "INSERT INTO stage_state (id, updated_at) VALUES (1, ?)",
            (db.utcnow_text(),),
        )


def _state_from_row(row):
    return {
        "speed_mm_s": row["speed_mm_s"],
        "step_mm": row["step_mm"],
        "workspace_mm": {
            "min_x": row["min_x"],
            "max_x": row["max_x"],
            "min_y": row["min_y"],
            "max_y": row["max_y"],
        },
        "machine_mm": {"x": row["machine_x"], "y": row["machine_y"]},
        "zero_mm": {"x": row["zero_x"], "y": row["zero_y"]},
        "zero_set": bool(row["zero_set"]),
        "points": json.loads(row["points_json"] or "[]"),
    }


def get_state():
    connection = db.connect()
    try:
        init_schema(connection)
        connection.commit()
        row = connection.execute("SELECT * FROM stage_state WHERE id = 1").fetchone()
        return _state_from_row(row)
    finally:
        connection.close()


def update_state(**fields):
    allowed = {
        "speed_mm_s",
        "step_mm",
        "min_x",
        "max_x",
        "min_y",
        "max_y",
        "machine_x",
        "machine_y",
        "zero_x",
        "zero_y",
        "zero_set",
        "points_json",
    }
    unknown = set(fields) - allowed
    if unknown:
        raise ValueError("Невідоме поле стану столу")
    if not fields:
        return get_state()
    assignments = ", ".join("%s = ?" % name for name in fields)
    values = list(fields.values()) + [db.utcnow_text()]
    connection = db.connect()
    try:
        init_schema(connection)
        connection.execute(
            "UPDATE stage_state SET %s, updated_at = ? WHERE id = 1" % assignments,
            values,
        )
        connection.commit()
        row = connection.execute("SELECT * FROM stage_state WHERE id = 1").fetchone()
        return _state_from_row(row)
    finally:
        connection.close()


def list_routes():
    connection = db.connect()
    try:
        init_schema(connection)
        connection.commit()
        rows = connection.execute(
            "SELECT id, name, speed_mm_s, points_json, updated_at FROM stage_routes ORDER BY name"
        ).fetchall()
        return [_route_from_row(row) for row in rows]
    finally:
        connection.close()


def get_route(route_id):
    connection = db.connect()
    try:
        row = connection.execute(
            "SELECT id, name, speed_mm_s, points_json, updated_at FROM stage_routes WHERE id = ?",
            (route_id,),
        ).fetchone()
        if row is None:
            return None
        return _route_from_row(row)
    finally:
        connection.close()


def save_route(name, points, speed_mm_s):
    payload = json.dumps(points, ensure_ascii=False)
    connection = db.connect()
    try:
        init_schema(connection)
        now = db.utcnow_text()
        existing = connection.execute(
            "SELECT id FROM stage_routes WHERE name = ?",
            (name,),
        ).fetchone()
        if existing is None:
            cursor = connection.execute(
                """
                INSERT INTO stage_routes (name, speed_mm_s, points_json, updated_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, speed_mm_s, payload, now),
            )
            route_id = cursor.lastrowid
        else:
            route_id = existing["id"]
            connection.execute(
                """
                UPDATE stage_routes
                SET speed_mm_s = ?, points_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (speed_mm_s, payload, now, route_id),
            )
        connection.commit()
        row = connection.execute(
            "SELECT id, name, speed_mm_s, points_json, updated_at FROM stage_routes WHERE id = ?",
            (route_id,),
        ).fetchone()
        return _route_from_row(row)
    finally:
        connection.close()


def delete_route(route_id):
    connection = db.connect()
    try:
        cursor = connection.execute("DELETE FROM stage_routes WHERE id = ?", (route_id,))
        connection.commit()
        return cursor.rowcount > 0
    finally:
        connection.close()


def _route_from_row(row):
    return {
        "id": row["id"],
        "name": row["name"],
        "speed_mm_s": row["speed_mm_s"],
        "points": json.loads(row["points_json"] or "[]"),
        "updated_at": row["updated_at"],
    }
