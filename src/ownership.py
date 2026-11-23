import json
import os

import auth
import db


def init_schema(connection):
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS results (
            run_id TEXT PRIMARY KEY,
            owner_user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS model_owners (
            model_type TEXT NOT NULL,
            model_name TEXT NOT NULL,
            owner_user_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (model_type, model_name)
        );
        """
    )


def _get_result_owner_from_legacy_metadata(hq_output_dir, run_id):
    metadata_path = os.path.join(hq_output_dir, run_id, "metadata.json")
    if not os.path.isfile(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as metadata_input:
            metadata = json.load(metadata_input)
    except Exception:
        return None
    owner_username = metadata.get("owner")
    if not owner_username:
        return None
    owner = auth.get_user_by_username(owner_username)
    return owner["id"] if owner else None


def get_result_owner_user_id(run_id):
    connection = db.connect()
    try:
        row = connection.execute("SELECT owner_user_id FROM results WHERE run_id = ?", (run_id,)).fetchone()
        return row["owner_user_id"] if row else None
    finally:
        connection.close()


def get_result_owner_username(hq_output_dir, run_id):
    owner_user_id = get_result_owner_user_id(run_id)
    if owner_user_id is not None:
        owner = auth.get_user_by_id(owner_user_id)
        if owner is not None:
            return owner["username"]
    metadata_path = os.path.join(hq_output_dir, run_id, "metadata.json")
    if not os.path.isfile(metadata_path):
        return "unknown"
    try:
        with open(metadata_path, "r", encoding="utf-8") as metadata_input:
            metadata = json.load(metadata_input)
    except Exception:
        return "unknown"
    return metadata.get("owner") or "unknown"


def store_result_owner(run_id, owner_user_id):
    connection = db.connect()
    try:
        connection.execute(
            """
            INSERT INTO results (run_id, owner_user_id, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(run_id) DO UPDATE SET
                owner_user_id = excluded.owner_user_id,
                created_at = excluded.created_at
            """,
            (run_id, owner_user_id, db.utcnow_text()),
        )
        connection.commit()
    finally:
        connection.close()


def delete_result_owner(run_id):
    connection = db.connect()
    try:
        connection.execute("DELETE FROM results WHERE run_id = ?", (run_id,))
        connection.commit()
    finally:
        connection.close()


def user_can_access_result(user, hq_output_dir, run_id):
    if auth.can_view_all_results(user):
        return True

    owner_user_id = get_result_owner_user_id(run_id)
    if owner_user_id is None:
        owner_user_id = _get_result_owner_from_legacy_metadata(hq_output_dir, run_id)
    if owner_user_id is None:
        return False
    return owner_user_id == user["id"]


def filter_results_for_user(user, hq_output_dir, results):
    if auth.can_view_all_results(user):
        return results
    return [item for item in results if user_can_access_result(user, hq_output_dir, item["id"])]


def get_model_owner_username(model_type, model_name):
    connection = db.connect()
    try:
        row = connection.execute(
            """
            SELECT users.username
            FROM model_owners
            JOIN users ON users.id = model_owners.owner_user_id
            WHERE model_owners.model_type = ? AND model_owners.model_name = ?
            """,
            (model_type, model_name),
        ).fetchone()
        return row["username"] if row else "system"
    finally:
        connection.close()


def store_model_owner(model_type, model_name, owner_user_id):
    connection = db.connect()
    try:
        connection.execute(
            """
            INSERT INTO model_owners (model_type, model_name, owner_user_id, created_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(model_type, model_name) DO UPDATE SET
                owner_user_id = excluded.owner_user_id,
                created_at = excluded.created_at
            """,
            (model_type, model_name, owner_user_id, db.utcnow_text()),
        )
        connection.commit()
    finally:
        connection.close()


def delete_model_owner(model_type, model_name):
    candidates = {model_name}
    if model_name.endswith("-fp16"):
        candidates.add(model_name[:-5])
    connection = db.connect()
    try:
        connection.executemany(
            "DELETE FROM model_owners WHERE model_type = ? AND model_name = ?",
            [(model_type, candidate) for candidate in candidates],
        )
        connection.commit()
    finally:
        connection.close()
