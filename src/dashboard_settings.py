import db


DEFAULT_DASHBOARD_SETTINGS = {
    "site_name": "",
    "analysis_number": "",
    "source_type": "camera",
    "camera_device": "",
    "camera_mode": {
        "width": 1280,
        "height": 720,
        "fps": 30,
        "format": "MJPG",
    },
    "uploaded_path": "",
    "model_path": "",
    "vis_conf": 0.75,
    "captions_enabled": True,
    "grid_count_enabled": False,
    "grid_debug_enabled": False,
    "grid_score_threshold": 0.30,
    "ask_manual_spore_count": True,
    "max_run_minutes": 30,
}


def init_schema(connection):
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS dashboard_settings (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            site_name TEXT NOT NULL DEFAULT '',
            analysis_number TEXT NOT NULL DEFAULT '',
            source_type TEXT NOT NULL DEFAULT 'camera',
            camera_device TEXT NOT NULL DEFAULT '',
            camera_width INTEGER NOT NULL DEFAULT 1280,
            camera_height INTEGER NOT NULL DEFAULT 720,
            camera_fps INTEGER NOT NULL DEFAULT 30,
            camera_format TEXT NOT NULL DEFAULT 'MJPG',
            uploaded_path TEXT NOT NULL DEFAULT '',
            model_path TEXT NOT NULL DEFAULT '',
            vis_conf REAL NOT NULL DEFAULT 0.75,
            captions_enabled INTEGER NOT NULL DEFAULT 1,
            grid_count_enabled INTEGER NOT NULL DEFAULT 0,
            grid_debug_enabled INTEGER NOT NULL DEFAULT 0,
            grid_score_threshold REAL NOT NULL DEFAULT 0.30,
            ask_manual_spore_count INTEGER NOT NULL DEFAULT 1,
            max_run_minutes INTEGER NOT NULL DEFAULT 30,
            updated_at TEXT NOT NULL
        );
        """
    )

    if not db.has_column(connection, "dashboard_settings", "ask_manual_spore_count"):
        connection.execute(
            "ALTER TABLE dashboard_settings ADD COLUMN ask_manual_spore_count INTEGER NOT NULL DEFAULT 1"
        )

    if not db.has_column(connection, "dashboard_settings", "captions_enabled"):
        connection.execute(
            "ALTER TABLE dashboard_settings ADD COLUMN captions_enabled INTEGER NOT NULL DEFAULT 1"
        )

    if not db.has_column(connection, "dashboard_settings", "site_name"):
        connection.execute(
            "ALTER TABLE dashboard_settings ADD COLUMN site_name TEXT NOT NULL DEFAULT ''"
        )

    if not db.has_column(connection, "dashboard_settings", "max_run_minutes"):
        connection.execute(
            "ALTER TABLE dashboard_settings ADD COLUMN max_run_minutes INTEGER NOT NULL DEFAULT 30"
        )

    if connection.execute("SELECT id FROM dashboard_settings WHERE id = 1").fetchone() is None:
        connection.execute(
            """
            INSERT INTO dashboard_settings (
                id,
                site_name,
                analysis_number,
                source_type,
                camera_device,
                camera_width,
                camera_height,
                camera_fps,
                camera_format,
                uploaded_path,
                model_path,
                vis_conf,
                captions_enabled,
                grid_count_enabled,
                grid_debug_enabled,
                grid_score_threshold,
                ask_manual_spore_count,
                max_run_minutes,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                DEFAULT_DASHBOARD_SETTINGS["site_name"],
                DEFAULT_DASHBOARD_SETTINGS["analysis_number"],
                DEFAULT_DASHBOARD_SETTINGS["source_type"],
                DEFAULT_DASHBOARD_SETTINGS["camera_device"],
                DEFAULT_DASHBOARD_SETTINGS["camera_mode"]["width"],
                DEFAULT_DASHBOARD_SETTINGS["camera_mode"]["height"],
                DEFAULT_DASHBOARD_SETTINGS["camera_mode"]["fps"],
                DEFAULT_DASHBOARD_SETTINGS["camera_mode"]["format"],
                DEFAULT_DASHBOARD_SETTINGS["uploaded_path"],
                DEFAULT_DASHBOARD_SETTINGS["model_path"],
                DEFAULT_DASHBOARD_SETTINGS["vis_conf"],
                int(DEFAULT_DASHBOARD_SETTINGS["captions_enabled"]),
                int(DEFAULT_DASHBOARD_SETTINGS["grid_count_enabled"]),
                int(DEFAULT_DASHBOARD_SETTINGS["grid_debug_enabled"]),
                DEFAULT_DASHBOARD_SETTINGS["grid_score_threshold"],
                int(DEFAULT_DASHBOARD_SETTINGS["ask_manual_spore_count"]),
                int(DEFAULT_DASHBOARD_SETTINGS["max_run_minutes"]),
                db.utcnow_text(),
            ),
        )


def get_dashboard_settings():
    connection = db.connect()
    try:
        row = connection.execute("SELECT * FROM dashboard_settings WHERE id = 1").fetchone()
        if row is None:
            return DEFAULT_DASHBOARD_SETTINGS.copy()
        return {
            "site_name": row["site_name"],
            "analysis_number": row["analysis_number"],
            "source_type": row["source_type"],
            "camera_device": row["camera_device"],
            "camera_mode": {
                "width": row["camera_width"],
                "height": row["camera_height"],
                "fps": row["camera_fps"],
                "format": row["camera_format"],
            },
            "uploaded_path": row["uploaded_path"],
            "model_path": row["model_path"],
            "vis_conf": float(row["vis_conf"]),
            "captions_enabled": bool(row["captions_enabled"]),
            "grid_count_enabled": bool(row["grid_count_enabled"]),
            "grid_debug_enabled": bool(row["grid_debug_enabled"]),
            "grid_score_threshold": float(row["grid_score_threshold"]),
            "ask_manual_spore_count": bool(row["ask_manual_spore_count"]),
            "max_run_minutes": int(row["max_run_minutes"]),
        }
    finally:
        connection.close()


def resolve_dashboard_start_payload(raw_payload, *, can_configure):
    settings = get_dashboard_settings()
    raw_payload = raw_payload or {}

    if not can_configure and raw_payload:
        raise PermissionError("Regular users cannot override dashboard settings.")

    mode = settings.get("camera_mode") or DEFAULT_DASHBOARD_SETTINGS["camera_mode"]
    return {
        "analysis_number": raw_payload.get("analysis_number", settings.get("analysis_number", "")),
        "source_type": raw_payload.get("source_type", settings.get("source_type", "camera")),
        "device": raw_payload.get("device", settings.get("camera_device", "")),
        "width": int(raw_payload.get("width", mode.get("width", 1280))),
        "height": int(raw_payload.get("height", mode.get("height", 720))),
        "fps": int(raw_payload.get("fps", mode.get("fps", 30))),
        "format": raw_payload.get("format", mode.get("format", "MJPG")),
        "video": raw_payload.get("video", settings.get("uploaded_path", "")),
        "model_path": raw_payload.get("model_path", settings.get("model_path", "")),
        "model_task": raw_payload.get("model_task"),
        "vis_conf": raw_payload.get("vis_conf", settings.get("vis_conf", 0.75)),
        "captions_enabled": raw_payload.get("captions_enabled", settings.get("captions_enabled", True)),
        "grid_count_enabled": raw_payload.get("grid_count_enabled", settings.get("grid_count_enabled", False)),
        "grid_debug_enabled": raw_payload.get("grid_debug_enabled", settings.get("grid_debug_enabled", False)),
        "grid_score_threshold": raw_payload.get("grid_score_threshold", settings.get("grid_score_threshold", 0.30)),
    }


def update_dashboard_settings(payload):
    current = get_dashboard_settings()
    updated = {
        "site_name": current["site_name"],
        "analysis_number": current["analysis_number"],
        "source_type": current["source_type"],
        "camera_device": current["camera_device"],
        "camera_mode": dict(current["camera_mode"]),
        "uploaded_path": current["uploaded_path"],
        "model_path": current["model_path"],
        "vis_conf": current["vis_conf"],
        "captions_enabled": current["captions_enabled"],
        "grid_count_enabled": current["grid_count_enabled"],
        "grid_debug_enabled": current["grid_debug_enabled"],
        "grid_score_threshold": current["grid_score_threshold"],
        "ask_manual_spore_count": current["ask_manual_spore_count"],
        "max_run_minutes": current["max_run_minutes"],
    }

    if "site_name" in payload:
        updated["site_name"] = str(payload.get("site_name") or "").strip()[:120]
    if "analysis_number" in payload:
        updated["analysis_number"] = str(payload.get("analysis_number") or "")
    if "source_type" in payload:
        source_type = str(payload.get("source_type") or "").strip().lower()
        if source_type not in ("camera", "file"):
            raise ValueError("Invalid source_type.")
        updated["source_type"] = source_type
    if "camera_device" in payload:
        updated["camera_device"] = str(payload.get("camera_device") or "")
    if "camera_mode" in payload and isinstance(payload.get("camera_mode"), dict):
        mode = payload["camera_mode"]
        updated["camera_mode"] = {
            "width": int(mode.get("width", updated["camera_mode"]["width"])),
            "height": int(mode.get("height", updated["camera_mode"]["height"])),
            "fps": int(mode.get("fps", updated["camera_mode"]["fps"])),
            "format": str(mode.get("format", updated["camera_mode"]["format"]) or "MJPG"),
        }
    if "uploaded_path" in payload:
        updated["uploaded_path"] = str(payload.get("uploaded_path") or "")
    if "model_path" in payload:
        updated["model_path"] = str(payload.get("model_path") or "")
    if "vis_conf" in payload:
        updated["vis_conf"] = float(payload.get("vis_conf"))
    if "captions_enabled" in payload:
        updated["captions_enabled"] = bool(payload.get("captions_enabled"))
    if "grid_count_enabled" in payload:
        updated["grid_count_enabled"] = bool(payload.get("grid_count_enabled"))
    if "grid_debug_enabled" in payload:
        updated["grid_debug_enabled"] = bool(payload.get("grid_debug_enabled"))
    if "grid_score_threshold" in payload:
        updated["grid_score_threshold"] = float(payload.get("grid_score_threshold"))
    if "ask_manual_spore_count" in payload:
        updated["ask_manual_spore_count"] = bool(payload.get("ask_manual_spore_count"))
    if "max_run_minutes" in payload:
        try:
            minutes = int(payload.get("max_run_minutes") or 0)
        except (TypeError, ValueError):
            minutes = DEFAULT_DASHBOARD_SETTINGS["max_run_minutes"]
        if minutes <= 0:
            minutes = DEFAULT_DASHBOARD_SETTINGS["max_run_minutes"]
        updated["max_run_minutes"] = max(1, min(480, minutes))

    connection = db.connect()
    try:
        connection.execute(
            """
            UPDATE dashboard_settings
            SET site_name = ?,
                analysis_number = ?,
                source_type = ?,
                camera_device = ?,
                camera_width = ?,
                camera_height = ?,
                camera_fps = ?,
                camera_format = ?,
                uploaded_path = ?,
                model_path = ?,
                vis_conf = ?,
                captions_enabled = ?,
                grid_count_enabled = ?,
                grid_debug_enabled = ?,
                grid_score_threshold = ?,
                ask_manual_spore_count = ?,
                max_run_minutes = ?,
                updated_at = ?
            WHERE id = 1
            """,
            (
                updated["site_name"],
                updated["analysis_number"],
                updated["source_type"],
                updated["camera_device"],
                updated["camera_mode"]["width"],
                updated["camera_mode"]["height"],
                updated["camera_mode"]["fps"],
                updated["camera_mode"]["format"],
                updated["uploaded_path"],
                updated["model_path"],
                updated["vis_conf"],
                int(updated["captions_enabled"]),
                int(updated["grid_count_enabled"]),
                int(updated["grid_debug_enabled"]),
                updated["grid_score_threshold"],
                int(updated["ask_manual_spore_count"]),
                int(updated["max_run_minutes"]),
                db.utcnow_text(),
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return updated
