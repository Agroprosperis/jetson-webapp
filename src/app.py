import argparse
import auth
import cookies
import csv
import cv2
import flask
import io
import json
import logging
import os
import re
import requests
import shutil
import socket
import struct
import threading
import time
import tokens
import uuid
import zipfile

from camera_manager import CameraManager
from datetime import datetime
from flasgger import Swagger
from inference_pipeline import StreamPipeline
from model_manager import (
    ModelConflictError,
    ModelManager,
    ModelManagerError,
    ModelNotFoundError,
    ModelValidationError,
)
from result_ids import generate_unique_result_run_id
from stream_readers import FileReader, V4L2StreamReader
from urllib.parse import quote

LOGGER = logging.getLogger("app")
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILEPATH = "/app/config.json"
HQ_OUTPUT_DIR = "/app/output_hq"
UPLOAD_DIR = "/upload"
VENDOR_DIR = "/opt/web/vendor"
RESULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
RESULT_VIDEO_EXTENSIONS = (".mkv", ".mp4", ".avi", ".mov")

app = flask.Flask(__name__, template_folder=".")

# --- SWAGGER CONFIGURATION ---
app.config['SWAGGER'] = {
    'title': 'Desktop Inference Pipeline API',
    'uiversion': 3,
    'specs_route': '/api/docs/'  # The URL where Swagger UI will be available
}

# Initialize Flasgger
swagger = Swagger(app)
auth.init_auth_storage()
model_manager = ModelManager(auth_module=auth, logger=LOGGER)


@app.after_request
def apply_pending_auth_cookies(response):
    if getattr(flask.g, "_skip_auth_cookie_refresh", False):
        return response

    pending = getattr(flask.g, "_auth_cookie_refresh", None)
    if pending is None:
        return response

    return cookies.set_auth_cookies(
        response,
        pending["access_token"],
        pending["refresh_token"],
        access_max_age=tokens.ACCESS_TOKEN_TTL_SECONDS,
        refresh_max_age=tokens.REFRESH_TOKEN_TTL_SECONDS,
    )

# Global runtime state
pipeline = None  # type: StreamPipeline | None
current_config = {}
last_error = None  # type: str | None


def _resolve_video_input_path(video):
    if not video or os.path.isabs(video):
        return video

    return os.path.join(UPLOAD_DIR, os.path.basename(video))
pipeline_id = None  # type: str | None
runtime_options = {
    "grid_count_enabled": False,
    "grid_debug_enabled": False,
    "grid_score": None,
    "grid_score_threshold": 0.30,
    "grid_auto_disabled": False,
}
runtime_options_lock = threading.Lock()


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _coerce_score(value):
    if value is None or value == "":
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return max(min(score, 1.0), 0.0)


def _coerce_threshold(value):
    if value is None or value == "":
        return 0.0
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(min(threshold, 1.0), 0.0)


def get_grid_count_enabled():
    with runtime_options_lock:
        return bool(runtime_options.get("grid_count_enabled", False))


def get_grid_score():
    with runtime_options_lock:
        return runtime_options.get("grid_score")


def get_grid_debug_enabled():
    with runtime_options_lock:
        return bool(runtime_options.get("grid_debug_enabled", False))


def get_grid_score_threshold():
    with runtime_options_lock:
        return float(runtime_options.get("grid_score_threshold", 0.30))


def set_grid_count_enabled(enabled, *, auto: bool = False):
    coerced = _coerce_bool(enabled)
    with runtime_options_lock:
        runtime_options["grid_count_enabled"] = coerced
        runtime_options["grid_auto_disabled"] = bool(auto and not coerced)
        return runtime_options["grid_count_enabled"]


def set_grid_score(score):
    coerced = _coerce_score(score)
    with runtime_options_lock:
        runtime_options["grid_score"] = coerced
        return runtime_options["grid_score"]


def set_grid_debug_enabled(enabled):
    coerced = _coerce_bool(enabled)
    with runtime_options_lock:
        runtime_options["grid_debug_enabled"] = coerced
        return runtime_options["grid_debug_enabled"]


def set_grid_score_threshold(value):
    coerced = _coerce_threshold(value)
    with runtime_options_lock:
        runtime_options["grid_score_threshold"] = coerced
        return runtime_options["grid_score_threshold"]


def get_runtime_options():
    with runtime_options_lock:
        return {
            "grid_count_enabled": bool(runtime_options.get("grid_count_enabled", False)),
            "grid_debug_enabled": bool(runtime_options.get("grid_debug_enabled", False)),
            "grid_score": runtime_options.get("grid_score"),
            "grid_score_threshold": float(runtime_options.get("grid_score_threshold", 0.30)),
            "grid_auto_disabled": bool(runtime_options.get("grid_auto_disabled", False)),
        }


def get_grid_api_state():
    options = get_runtime_options()
    return {
        "enabled": options["grid_count_enabled"],
        "debug_enabled": options["grid_debug_enabled"],
        "score": options["grid_score"],
        "score_threshold": options["grid_score_threshold"],
        "auto_disabled": options["grid_auto_disabled"],
    }


def apply_grid_api_update(payload):
    if not isinstance(payload, dict):
        return False

    updated = False

    if "enabled" in payload:
        set_grid_count_enabled(payload.get("enabled"))
        updated = True

    if "debug_enabled" in payload:
        set_grid_debug_enabled(payload.get("debug_enabled"))
        updated = True

    if "score_threshold" in payload:
        set_grid_score_threshold(payload.get("score_threshold"))
        updated = True

    return updated


def is_pipeline_running():
    return any(alive for _, alive in pipeline_thread_states())


def pipeline_thread_states():
    """Return ordered pipeline thread labels and liveness."""
    global pipeline
    thread_specs = [
        ("capture", "capture_t"),
        ("inference", "inference_t"),
        ("grid", "grid_t"),
        ("output", "output_t"),
    ]
    if pipeline is None:
        return [(name, False) for name, _ in thread_specs]

    states: list[tuple[str, bool]] = []
    for name, attr in thread_specs:
        thread = getattr(pipeline, attr, None)
        if thread is not None:
            states.append((name, thread.is_alive()))
    if states:
        return states

    threads = getattr(pipeline, "_threads", None) or []
    fallback_names = [name for name, _ in thread_specs]
    states = []
    for idx, thread in enumerate(threads):
        label = fallback_names[idx] if idx < len(fallback_names) else f"thread_{idx}"
        states.append((label, thread.is_alive()))
    return states


def extract_id_or_date(filename):
    pattern = r"^(?:(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})-[a-fA-F0-9]{32}|(.+))\.mkv$"
    match = re.match(pattern, filename)
    
    if match:
        return match.group(1) or match.group(2)
    return None


def _parse_iso_date(value, field_name):
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(f"Invalid {field_name}. Use YYYY-MM-DD.") from exc


def _parse_results_date_filter():
    return _parse_iso_date(flask.request.args.get("date", "").strip(), "date")


def _result_matches_date(timestamp_value, selected_date=None):
    if not timestamp_value:
        return selected_date is None

    result_date = datetime.fromtimestamp(timestamp_value).date()
    return selected_date is None or result_date == selected_date


def _get_video_duration_seconds(video_path):
    if os.path.splitext(video_path)[1].lower() != ".mkv":
        return None

    return _get_mkv_duration_seconds(video_path)


def _read_ebml_vint(data, offset, *, strip_marker):
    if offset >= len(data):
        return None, offset

    first = data[offset]
    mask = 0x80
    length = 1
    while length <= 8 and not (first & mask):
        mask >>= 1
        length += 1

    if length > 8 or offset + length > len(data):
        return None, offset

    value = first & (mask - 1) if strip_marker else first
    for byte in data[offset + 1:offset + length]:
        value = (value << 8) | byte

    return value, offset + length


def _parse_ebml_float(payload):
    if len(payload) == 4:
        return float(struct.unpack(">f", payload)[0])
    if len(payload) == 8:
        return float(struct.unpack(">d", payload)[0])
    return None


def _parse_ebml_uint(payload):
    value = 0
    for byte in payload:
        value = (value << 8) | byte
    return value


def _get_mkv_duration_seconds(video_path, *, max_header_bytes=2_000_000):
    try:
        with open(video_path, "rb") as video_input:
            data = video_input.read(max_header_bytes)
    except OSError:
        LOGGER.warning("Failed to read MKV header: %s", video_path, exc_info=True)
        return None

    duration_marker = bytes.fromhex("4489")
    duration_index = data.find(duration_marker)
    if duration_index < 0:
        return None

    duration_id, value_offset = _read_ebml_vint(data, duration_index, strip_marker=False)
    if duration_id != 0x4489:
        return None

    duration_size, payload_offset = _read_ebml_vint(data, value_offset, strip_marker=True)
    if duration_size is None or duration_size <= 0:
        return None

    duration_end = payload_offset + duration_size
    if duration_end > len(data):
        return None

    duration_value = _parse_ebml_float(data[payload_offset:duration_end])
    if duration_value is None or duration_value <= 0:
        return None

    timecode_scale = 1_000_000
    scale_marker = bytes.fromhex("2AD7B1")
    scale_index = data.rfind(scale_marker, 0, duration_index)
    if scale_index >= 0:
        scale_id, scale_value_offset = _read_ebml_vint(data, scale_index, strip_marker=False)
        if scale_id == 0x2AD7B1:
            scale_size, scale_payload_offset = _read_ebml_vint(data, scale_value_offset, strip_marker=True)
            scale_end = scale_payload_offset + scale_size if scale_size is not None else len(data) + 1
            if scale_size is not None and scale_size > 0 and scale_end <= len(data):
                parsed_scale = _parse_ebml_uint(data[scale_payload_offset:scale_end])
                if parsed_scale > 0:
                    timecode_scale = parsed_scale

    return duration_value * timecode_scale / 1_000_000_000


def _read_last_nonempty_line(path, *, chunk_size=8192):
    with open(path, "rb") as input_file:
        input_file.seek(0, os.SEEK_END)
        position = input_file.tell()
        buffer = b""

        while position > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            input_file.seek(position)
            buffer = input_file.read(read_size) + buffer
            lines = buffer.splitlines()
            candidates = lines[1:] if position > 0 else lines

            for line in reversed(candidates):
                if line.strip():
                    return line.decode("utf-8")

    return None


def _read_last_csv_row_fast(csv_path):
    last_line = _read_last_nonempty_line(csv_path)
    if not last_line:
        return None

    with open(csv_path, "r", newline="", encoding="utf-8") as csv_input:
        header_line = csv_input.readline()
    if not header_line:
        return None

    reader = csv.DictReader(io.StringIO(header_line + last_line + "\n"))
    return next(reader, None)


def _format_duration(seconds):
    if seconds is None:
        return "-"

    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _read_result_csv_summary(csv_path):
    if not os.path.isfile(csv_path):
        return {
            "detected_objects": None,
            "class_counts": None,
            "s_value": None,
        }

    try:
        row = _coerce_csv_row(_read_last_csv_row_fast(csv_path))
    except Exception:
        LOGGER.exception("Failed to read result CSV summary: %s", csv_path)
        return {
            "detected_objects": None,
            "class_counts": None,
            "s_value": None,
        }

    if not row:
        return {
            "detected_objects": None,
            "class_counts": None,
            "s_value": None,
        }

    return {
        "detected_objects": row.get("tilletia_objects"),
        "class_counts": row.get("class_counts"),
        "s_value": row.get("s_value"),
    }


def _build_result_metadata(run_id):
    run_path = os.path.join(HQ_OUTPUT_DIR, run_id)
    if not os.path.isdir(run_path):
        return None

    files = []
    latest_mtime = 0
    longest_duration_seconds = None
    for filename in sorted(os.listdir(run_path)):
        full_path = os.path.join(run_path, filename)
        if not os.path.isfile(full_path):
            continue

        stat = os.stat(full_path)
        latest_mtime = max(latest_mtime, stat.st_mtime)
        size_mb = round(stat.st_size / (1024 * 1024), 2)
        files.append({
            "name": filename,
            "path": f"{run_id}/{filename}",
            "size": f"{size_mb} MB",
        })
        if os.path.splitext(filename)[1].lower() in RESULT_VIDEO_EXTENSIONS:
            duration_seconds = _get_video_duration_seconds(full_path)
            if duration_seconds is not None and (
                longest_duration_seconds is None
                or duration_seconds > longest_duration_seconds
            ):
                longest_duration_seconds = duration_seconds

    timestamp = datetime.fromtimestamp(latest_mtime).strftime('%Y-%m-%d %H:%M:%S') if latest_mtime else "-"
    csv_path = os.path.join(run_path, f"{run_id}.csv")
    video_path = os.path.join(run_path, f"{run_id}.mkv")
    csv_summary = _read_result_csv_summary(csv_path)

    return {
        "id": run_id,
        "run_path": run_path,
        "timestamp": timestamp,
        "duration": _format_duration(longest_duration_seconds),
        "duration_seconds": longest_duration_seconds,
        "detected_objects": csv_summary["detected_objects"],
        "class_counts": csv_summary.get("class_counts"),
        "s_value": csv_summary["s_value"],
        "files": files,
        "_mtime": latest_mtime,
        "csv_path": csv_path if os.path.isfile(csv_path) else None,
        "video_path": video_path if os.path.isfile(video_path) else None,
    }


def _build_download_url(host_url, relative_path):
    return f"{host_url}/download/{quote(relative_path, safe='/')}"


def _resolve_result_video_path(run_id, requested_relative_path=""):
    run_dir = os.path.abspath(os.path.join(HQ_OUTPUT_DIR, run_id))
    if requested_relative_path:
        normalized_relative_path = str(requested_relative_path).replace("\\", "/").lstrip("/")
        if normalized_relative_path.split("/", 1)[0] != run_id:
            raise ValueError("Invalid result video path.")
        candidate_path = os.path.abspath(os.path.join(HQ_OUTPUT_DIR, normalized_relative_path))
        if candidate_path != run_dir and not candidate_path.startswith(run_dir + os.sep):
            raise ValueError("Invalid result video path.")
    else:
        candidate_path = os.path.abspath(os.path.join(run_dir, f"{run_id}.mkv"))

    if os.path.splitext(candidate_path)[1].lower() not in RESULT_VIDEO_EXTENSIONS:
        raise ValueError("Selected result file is not a video.")
    if not os.path.isfile(candidate_path):
        raise FileNotFoundError("Result video not found.")
    return candidate_path


def _is_result_image(filename):
    return os.path.splitext(filename)[1].lower() in RESULT_IMAGE_EXTENSIONS


def _collect_results_metadata(*, analysis_prefix="", selected_date=None):
    if not os.path.exists(HQ_OUTPUT_DIR):
        return []

    results = []
    run_dirs = [
        name for name in os.listdir(HQ_OUTPUT_DIR)
        if os.path.isdir(os.path.join(HQ_OUTPUT_DIR, name))
    ]

    for run_id in run_dirs:
        if analysis_prefix and analysis_prefix not in run_id:
            continue

        metadata = _build_result_metadata(run_id)
        if metadata is None:
            continue
        if not _result_matches_date(metadata["_mtime"], selected_date):
            continue

        results.append(metadata)

    results.sort(key=lambda item: item["_mtime"], reverse=True)
    return results


def _read_last_csv_row(csv_path):
    last_row = None
    with open(csv_path, "r", newline="", encoding="utf-8") as csv_input:
        reader = csv.DictReader(csv_input)
        for row in reader:
            if not row:
                continue
            if not any((value or "").strip() for value in row.values()):
                continue
            last_row = row
    return last_row


def _parse_csv_detections(value):
    if not value or not value.strip():
        return []

    try:
        detections = json.loads(value)
    except json.JSONDecodeError:
        LOGGER.warning("Skipping malformed CSV detections JSON: %s", value)
        return []

    return detections if isinstance(detections, list) else []


def _parse_csv_class_counts(value):
    if not value or not value.strip():
        return {}

    try:
        class_counts = json.loads(value)
    except json.JSONDecodeError:
        LOGGER.warning("Skipping malformed CSV class_counts JSON: %s", value)
        return {}

    if not isinstance(class_counts, dict):
        return {}

    parsed = {}
    for class_name, count in class_counts.items():
        try:
            parsed[str(class_name)] = int(count)
        except (TypeError, ValueError):
            continue
    return parsed


def _coerce_csv_row(row):
    if not row:
        return None

    coerced = dict(row)
    for key in ("frame", "tilletia_objects"):
        value = coerced.get(key)
        try:
            coerced[key] = int(value)
        except (TypeError, ValueError):
            pass

    value = coerced.get("s_value")
    try:
        coerced["s_value"] = float(value)
    except (TypeError, ValueError):
        pass

    coerced["detections"] = _parse_csv_detections(coerced.get("detections"))
    coerced["class_counts"] = _parse_csv_class_counts(coerced.get("class_counts"))

    return coerced


@app.route("/login")
def login_page():
    """
    Serve the login page.
    ---
    tags:
      - Pages
    produces:
      - text/html
    responses:
      200:
        description: Login page HTML.
      302:
        description: Redirect to the dashboard when the caller is already authenticated.
    """
    user = auth.load_request_user()
    if user is not None:
        return flask.redirect(flask.url_for("index"))
    return flask.render_template("login.html")


@app.route("/auth-client.js")
def auth_client_asset():
    return flask.send_from_directory(SRC_DIR, "auth_client.js", mimetype="application/javascript")


@app.route("/auth/login", methods=["POST"])
def auth_login():
    """
    Authenticate a local user and issue bearer tokens.
    ---
    tags:
      - Auth
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - username
            - password
          properties:
            username:
              type: string
            password:
              type: string
    responses:
      200:
        description: Access token, refresh token, and authenticated user identity.
        schema:
          type: object
          properties:
            access_token:
              type: string
            refresh_token:
              type: string
            user:
              type: object
              properties:
                id:
                  type: integer
                username:
                  type: string
      401:
        description: Invalid username or password.
      403:
        description: Password change is required before first login.
    """
    payload = flask.request.get_json(silent=True) or {}
    user = auth.authenticate_user(payload.get("username", ""), payload.get("password", ""))
    if user is None:
        return flask.jsonify({"error": "Invalid username or password"}), 401
    if user.get("force_password_change"):
        return (
            flask.jsonify(
                {
                    "error": "Password change required.",
                    "password_change_required": True,
                }
            ),
            403,
        )

    access_token = tokens.issue_access_token(user["id"])
    refresh_token = auth.issue_refresh_token(user["id"])
    response = flask.jsonify(
        {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user": {
                "id": user["id"],
                "username": user["username"],
            },
        }
    )
    return cookies.set_auth_cookies(
        response,
        access_token,
        refresh_token,
        access_max_age=tokens.ACCESS_TOKEN_TTL_SECONDS,
        refresh_max_age=tokens.REFRESH_TOKEN_TTL_SECONDS,
    )


@app.route("/auth/refresh", methods=["POST"])
def auth_refresh():
    """
    Exchange a refresh token for a new bearer access token.
    ---
    tags:
      - Auth
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: false
        schema:
          type: object
          properties:
            refresh_token:
              type: string
              description: Optional refresh token. When omitted, the refresh cookie is used.
    responses:
      200:
        description: Refreshed access token, rotated refresh token, and authenticated user identity.
        schema:
          type: object
          properties:
            access_token:
              type: string
            refresh_token:
              type: string
            user:
              type: object
              properties:
                id:
                  type: integer
                username:
                  type: string
      401:
        description: Missing or invalid refresh token.
    """
    payload = flask.request.get_json(silent=True) or {}
    refresh_token = (payload.get("refresh_token") or cookies.get_refresh_token_from_request(flask.request)).strip()
    if not refresh_token:
        return flask.jsonify({"error": "Missing refresh token"}), 401

    rotated = auth.rotate_refresh_token(refresh_token)
    if rotated is None:
        return flask.jsonify({"error": "Invalid refresh token"}), 401

    user = rotated["user"]
    access_token = tokens.issue_access_token(user["id"])
    response = flask.jsonify(
        {
            "access_token": access_token,
            "refresh_token": rotated["refresh_token"],
            "user": {
                "id": user["id"],
                "username": user["username"],
            },
        }
    )
    return cookies.set_auth_cookies(
        response,
        access_token,
        rotated["refresh_token"],
        access_max_age=tokens.ACCESS_TOKEN_TTL_SECONDS,
        refresh_max_age=tokens.REFRESH_TOKEN_TTL_SECONDS,
    )


@app.route("/auth/change-password", methods=["POST"])
def auth_change_password():
    """
    Change a local user's password and clear the first-login password-change requirement.
    ---
    tags:
      - Auth
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - username
            - current_password
            - new_password
            - confirm_password
          properties:
            username:
              type: string
            current_password:
              type: string
            new_password:
              type: string
            confirm_password:
              type: string
    responses:
      200:
        description: Password updated successfully.
        schema:
          type: object
          properties:
            success:
              type: boolean
      400:
        description: Invalid password-change payload.
      401:
        description: Invalid username or current password.
    """
    payload = flask.request.get_json(silent=True) or {}
    if payload.get("new_password") != payload.get("confirm_password"):
        return flask.jsonify({"error": "Passwords do not match"}), 400
    try:
        user = auth.change_password(
            payload.get("username"),
            payload.get("current_password"),
            payload.get("new_password"),
        )
    except ValueError as exc:
        return flask.jsonify({"error": str(exc)}), 400
    if user is None:
        return flask.jsonify({"error": "Invalid username or current password"}), 401
    return flask.jsonify({"success": True})


@app.route("/auth/logout", methods=["POST"])
@auth.require_permission()
def auth_logout():
    """
    Revoke the current refresh token and clear auth cookies.
    ---
    tags:
      - Auth
    responses:
      200:
        description: Logout completed successfully.
        schema:
          type: object
          properties:
            success:
              type: boolean
      401:
        description: Missing or invalid access token.
    """
    flask.g._skip_auth_cookie_refresh = True
    auth.revoke_refresh_token(cookies.get_refresh_token_from_request(flask.request))
    response = flask.jsonify({"success": True})
    return cookies.clear_auth_cookies(response)


@app.route("/auth/me")
@auth.require_permission()
def auth_me():
    """
    Return the authenticated user identity.
    ---
    tags:
      - Auth
    responses:
      200:
        description: Authenticated user identity.
        schema:
          type: object
          properties:
            id:
              type: integer
            username:
              type: string
      401:
        description: Missing or invalid access token.
    """
    user = flask.g.current_user
    return flask.jsonify(
        {
            "id": user["id"],
            "username": user["username"],
        }
    )


@app.route("/users", methods=["POST"])
@auth.require_permission("users:manage")
def create_user_route():
    """
    Create a local user with one or more roles.
    ---
    tags:
      - Auth
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - username
            - password
            - roles
          properties:
            username:
              type: string
            password:
              type: string
            roles:
              type: array
              items:
                type: string
                enum: ['admin', 'user']
    responses:
      201:
        description: Created user summary.
        schema:
          type: object
          properties:
            id:
              type: integer
            username:
              type: string
      400:
        description: Invalid payload.
      401:
        description: Missing or invalid access token.
      403:
        description: Authenticated user lacks permission to manage users.
      409:
        description: Username already exists.
    """
    payload = flask.request.get_json(silent=True) or {}
    try:
        user = auth.create_user(
            payload.get("username"),
            payload.get("password"),
            payload.get("roles"),
        )
    except ValueError as exc:
        message = str(exc)
        status = 409 if "exists" in message.lower() else 400
        return flask.jsonify({"error": message}), status

    return (
        flask.jsonify(
            {
                "id": user["id"],
                "username": user["username"],
            }
        ),
        201,
    )


@app.route("/users")
@auth.require_permission("users:manage", html_redirect=True)
def users_page():
    """
    Serve the user-management page.
    ---
    tags:
      - Pages
    produces:
      - text/html
    responses:
      200:
        description: User-management HTML.
      302:
        description: Redirect to the login page when the caller is not authenticated.
      403:
        description: Authenticated user lacks permission to manage users.
    """
    return flask.render_template(
        "users.html",
        **auth.build_page_context(flask.g.current_user, users=auth.list_users()),
    )


@app.route("/settings")
@auth.require_permission(html_redirect=True)
def settings_page():
    """
    Serve the authenticated user settings page.
    ---
    tags:
      - Pages
    produces:
      - text/html
    responses:
      200:
        description: User settings HTML.
      302:
        description: Redirect to the login page when the caller is not authenticated.
      401:
        description: Missing or invalid access token.
    """
    return flask.render_template(
        "settings.html",
        role_label=", ".join(flask.g.current_user.get("roles", [])),
        **auth.build_page_context(flask.g.current_user),
    )


@app.route("/")
@auth.require_permission("dashboard:view", html_redirect=True)
def index():
    """
    Serve the main dashboard page.
    ---
    tags:
      - Pages
    produces:
      - text/html
    responses:
      200:
        description: Dashboard HTML.
      302:
        description: Redirect to the login page when the caller is not authenticated.
      403:
        description: Authenticated user lacks permission to view the dashboard.
    """
    return flask.render_template("index.html", **auth.build_page_context(flask.g.current_user))


@app.route("/results")
@auth.require_permission("results:view", html_redirect=True)
def results_page():
    """
    Serve the results page.
    ---
    tags:
      - Pages
    produces:
      - text/html
    responses:
      200:
        description: Results page HTML.
      302:
        description: Redirect to the login page when the caller is not authenticated.
      403:
        description: Authenticated user lacks permission to view results.
    """
    return flask.render_template("results.html", **auth.build_page_context(flask.g.current_user))


@app.route("/models")
@auth.require_permission("models:view", html_redirect=True)
def models_page():
    """
    Serve the model catalog page.
    ---
    tags:
      - Pages
    produces:
      - text/html
    responses:
      200:
        description: Model catalog HTML.
      302:
        description: Redirect to the login page when the caller is not authenticated.
      403:
        description: Authenticated user lacks permission to view the model catalog.
    """
    return flask.render_template("models.html", **auth.build_page_context(flask.g.current_user))


@app.route("/api/config")
@auth.require_permission("dashboard:view")
def api_config():
    """
    Get basic runtime configuration.
    ---
    tags:
      - Configuration
    responses:
      200:
        description: Basic configuration values
        schema:
          type: object
          properties:
            stream_port:
              type: integer
    """
    return flask.jsonify({"stream_port": 8889})


@app.route("/api/dashboard-settings", methods=["GET"])
@auth.require_permission("dashboard_settings:view")
def api_get_dashboard_settings():
    """
    Get the persisted dashboard settings used as defaults for new runs.
    ---
    tags:
      - Configuration
    responses:
      200:
        description: Current dashboard settings.
        schema:
          type: object
          properties:
            analysis_number:
              type: string
            source_type:
              type: string
              enum: ['camera', 'file']
            camera_device:
              type: string
            camera_mode:
              type: object
              properties:
                width:
                  type: integer
                height:
                  type: integer
                fps:
                  type: integer
                format:
                  type: string
            uploaded_path:
              type: string
            model_path:
              type: string
            vis_conf:
              type: number
            grid_count_enabled:
              type: boolean
            grid_debug_enabled:
              type: boolean
            grid_score_threshold:
              type: number
      401:
        description: Missing or invalid access token.
      403:
        description: Authenticated user lacks permission to view dashboard settings.
    """
    return flask.jsonify(auth.get_dashboard_settings())


@app.route("/api/dashboard-settings", methods=["PUT"])
@auth.require_permission("dashboard:configure")
def api_put_dashboard_settings():
    """
    Update the persisted dashboard settings used as defaults for new runs.
    ---
    tags:
      - Configuration
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            analysis_number:
              type: string
            source_type:
              type: string
              enum: ['camera', 'file']
            camera_device:
              type: string
            camera_mode:
              type: object
              properties:
                width:
                  type: integer
                height:
                  type: integer
                fps:
                  type: integer
                format:
                  type: string
            uploaded_path:
              type: string
            model_path:
              type: string
            vis_conf:
              type: number
            grid_count_enabled:
              type: boolean
            grid_debug_enabled:
              type: boolean
            grid_score_threshold:
              type: number
    responses:
      200:
        description: Updated dashboard settings.
      400:
        description: Invalid settings payload.
      401:
        description: Missing or invalid access token.
      403:
        description: Authenticated user lacks permission to update dashboard settings.
    """
    payload = flask.request.get_json(silent=True) or {}
    try:
        settings = auth.update_dashboard_settings(payload)
    except (TypeError, ValueError) as exc:
        return flask.jsonify({"error": str(exc)}), 400
    return flask.jsonify(settings)


@app.route("/api/grid", methods=["GET"])
@auth.require_permission("dashboard:view")
def api_get_grid():
    """
    Get the current grid feature state.
    ---
    tags:
      - Grid
    responses:
      200:
        description: Current grid feature state
        schema:
          type: object
          properties:
            enabled:
              type: boolean
              description: True when grid detection and viewport-based counting are enabled.
            debug_enabled:
              type: boolean
              description: True when the rich grid debug overlay is rendered on the output stream.
            score:
              type: number
              description: Current EMA grid quality score in the 0..1 range, or null before the first processed frame.
            score_threshold:
              type: number
              description: EMA score threshold that auto-disables the grid feature. 0 disables auto-off.
            auto_disabled:
              type: boolean
              description: True when the feature was switched off automatically because the EMA score dropped below the threshold.
    """
    return flask.jsonify(get_grid_api_state())


@app.route("/api/grid", methods=["PUT"])
@auth.require_permission("dashboard:configure")
def api_put_grid():
    """
    Update grid feature settings.
    ---
    tags:
      - Grid
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          additionalProperties: false
          properties:
            enabled:
              type: boolean
              description: Enable grid detection, grid overlay, and viewport-based counting.
            debug_enabled:
              type: boolean
              description: Enable the rich grid debug overlay with raw, accumulated, rejected, predicted, and accepted lines.
            score_threshold:
              type: number
              minimum: 0
              maximum: 1
              description: EMA grid score threshold that auto-disables the grid feature. Use 0 to disable auto-off.
    responses:
      200:
        description: Updated grid feature state
        schema:
          type: object
          properties:
            enabled:
              type: boolean
            debug_enabled:
              type: boolean
            score:
              type: number
            score_threshold:
              type: number
            auto_disabled:
              type: boolean
      400:
        description: Missing grid settings in request body
    """
    payload = flask.request.get_json(silent=True) or {}
    if not apply_grid_api_update(payload):
        return flask.jsonify({"error": "Missing grid settings"}), 400

    update_payload = {}
    if "enabled" in payload:
        update_payload["grid_count_enabled"] = payload.get("enabled")
    if "debug_enabled" in payload:
        update_payload["grid_debug_enabled"] = payload.get("debug_enabled")
    if "score_threshold" in payload:
        update_payload["grid_score_threshold"] = payload.get("score_threshold")
    if update_payload:
        try:
            auth.update_dashboard_settings(update_payload)
        except (TypeError, ValueError):
            pass

    return flask.jsonify(get_grid_api_state())


@app.route("/api/models")
@auth.require_permission("models:view")
def api_models():
    """
    List available *.engine models.
    ---
    tags:
      - Configuration
    responses:
      200:
        description: List of available TensorRT engine files
        schema:
          type: object
          properties:
            models:
              type: array
              items:
                type: object
                properties:
                  name:
                    type: string
                  type:
                    type: string
                  path:
                    type: string
                  display:
                    type: string
                  default_confidence_threshold:
                    type: number
                  owner_username:
                    type: string
    """
    try:
        return flask.jsonify({"models": model_manager.list_engine_models()})
    except Exception as e:
        LOGGER.error(f"Failed to list models: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/api/model-catalog")
@auth.require_permission("models:view")
def api_model_catalog():
    """
    List available model sources and their TensorRT engines (if compiled).
    ---
    tags:
      - Configuration
    responses:
      200:
        description: List of model files and their TensorRT engines
        schema:
          type: object
          properties:
            models:
              type: array
              items:
                type: object
                properties:
                  name:
                    type: string
                  type:
                    type: string
                  sources:
                    type: array
                    items:
                      type: string
                  source_paths:
                    type: array
                    items:
                      type: string
                  engine:
                    type: object
                    properties:
                      name:
                        type: string
                      path:
                        type: string
                  task:
                    type: string
                    enum: ['segment', 'detect', 'auto']
                    description: Saved Ultralytics task override for this model. Null for non-UL models.
                  default_confidence_threshold:
                    type: number
                    description: Saved per-model default confidence threshold used by the main dashboard when this model is selected.
                  compiled:
                    type: boolean
                  owner_username:
                    type: string
            tensorrt:
              type: object
              properties:
                current:
                  type: string
    """
    try:
        return flask.jsonify(
            {
                "models": model_manager.build_model_catalog(),
                "tensorrt": {"current": model_manager.get_current_tensorrt_version()},
            }
        )
    except Exception as e:
        LOGGER.error(f"Failed to list model catalog: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/api/model-upload", methods=["POST"])
@auth.require_permission("models:manage")
def api_model_upload():
    """
    Upload a model weights file or RF deployment package into the catalog.
    ---
    tags:
      - Configuration
    consumes:
      - multipart/form-data
    parameters:
      - name: type
        in: formData
        type: string
        required: true
        enum: ['ul', 'rf']
        description: Target model family for the uploaded weights file.
      - name: file
        in: formData
        type: file
        required: true
        description: Ultralytics/RF-DETR `.pt` weights, or an RF deployment package `.zip`.
    responses:
      200:
        description: Model uploaded successfully
        schema:
          type: object
          properties:
            uploaded:
              type: object
              properties:
                type:
                  type: string
                name:
                  type: string
                path:
                  type: string
            tensorrt:
              type: object
              properties:
                current:
                  type: string
      400:
        description: Invalid upload request
      409:
        description: A model with the same filename already exists.
    """
    model_type = flask.request.form.get("type")
    uploaded_file = flask.request.files.get("file")

    try:
        uploaded = model_manager.upload_model(
            model_type,
            uploaded_file,
            flask.g.current_user["id"],
        )
        return flask.jsonify(
            {
                "uploaded": uploaded,
                "tensorrt": {"current": model_manager.get_current_tensorrt_version()},
            }
        )
    except ModelValidationError as exc:
        return flask.jsonify({"error": str(exc)}), 400
    except ModelConflictError as exc:
        return flask.jsonify({"error": str(exc)}), 409
    except ModelManagerError as exc:
        return flask.jsonify({"error": str(exc)}), 500


@app.route("/api/model-compile", methods=["POST"])
@auth.require_permission("models:manage")
def api_model_compile():
    """
    Start async model compilation to TensorRT (FP16).
    ---
    tags:
      - Configuration
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - type
            - name
          properties:
            type:
              type: string
              enum: ['ul', 'rf']
            name:
              type: string
            inference_width:
              type: integer
              minimum: 32
              description: Optional compile width for Ultralytics engine export.
            inference_height:
              type: integer
              minimum: 32
              description: Optional compile height for Ultralytics engine export.
    responses:
      200:
        description: Compile job queued
        schema:
          type: object
          properties:
            job_id:
              type: string
            already_running:
              type: boolean
      400:
        description: Invalid compile request or model is already compiled.
      404:
        description: Model not found.
    """
    payload = flask.request.get_json(silent=True) or {}

    try:
        return flask.jsonify(
            model_manager.start_compile(
                payload.get("type"),
                payload.get("name"),
                inference_width=payload.get("inference_width"),
                inference_height=payload.get("inference_height"),
            )
        )
    except ModelValidationError as exc:
        return flask.jsonify({"error": str(exc)}), 400
    except ModelNotFoundError as exc:
        return flask.jsonify({"error": str(exc)}), 404


@app.route("/api/model-task", methods=["POST"])
@auth.require_permission("models:manage")
def api_model_task():
    """
    Save the Ultralytics task override for a catalog model.
    ---
    tags:
      - Configuration
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - type
            - name
            - task
          properties:
            type:
              type: string
              enum: ['ul']
              description: Model family. Only Ultralytics models support task overrides.
            name:
              type: string
              description: Catalog model base name.
            task:
              type: string
              enum: ['segment', 'detect', 'auto']
              description: Ultralytics task to use when loading the model.
    responses:
      200:
        description: Saved task override
        schema:
          type: object
          properties:
            type:
              type: string
            name:
              type: string
            task:
              type: string
              enum: ['segment', 'detect', 'auto']
      400:
        description: Invalid payload or task
      404:
        description: Model not found
    """
    payload = flask.request.get_json(silent=True) or {}

    try:
        return flask.jsonify(
            model_manager.set_model_task(
                payload.get("type"),
                payload.get("name"),
                payload.get("task"),
            )
        )
    except ModelValidationError as exc:
        return flask.jsonify({"error": str(exc)}), 400
    except ModelNotFoundError as exc:
        return flask.jsonify({"error": str(exc)}), 404


@app.route("/api/model-metadata", methods=["POST"])
@auth.require_permission("models:manage")
def api_model_metadata():
    """
    Save editable metadata for a catalog model.
    ---
    tags:
      - Configuration
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - type
            - name
          properties:
            type:
              type: string
              enum: ['ul', 'rf']
            name:
              type: string
            default_confidence_threshold:
              type: number
              minimum: 0
              maximum: 1
              description: Saved per-model default confidence threshold for the main dashboard.
    responses:
      200:
        description: Saved model metadata
      400:
        description: Invalid payload
      404:
        description: Model not found
    """
    payload = flask.request.get_json(silent=True) or {}

    try:
        return flask.jsonify(model_manager.set_model_metadata(payload.get("type"), payload.get("name"), payload))
    except ModelValidationError as exc:
        return flask.jsonify({"error": str(exc)}), 400
    except ModelNotFoundError as exc:
        return flask.jsonify({"error": str(exc)}), 404


@app.route("/api/model-delete", methods=["POST"])
@auth.require_permission("models:manage")
def api_model_delete():
    """
    Delete all stored artifacts for a catalog model.
    ---
    tags:
      - Configuration
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - type
            - name
          properties:
            type:
              type: string
              enum: ['ul', 'rf']
              description: Model family.
            name:
              type: string
              description: Catalog model base name.
    responses:
      200:
        description: Deleted artifact list
        schema:
          type: object
          properties:
            type:
              type: string
            name:
              type: string
            deleted:
              type: array
              items:
                type: string
      400:
        description: Invalid payload
      404:
        description: Model not found
      409:
        description: Model is currently running and cannot be deleted
    """
    payload = flask.request.get_json(silent=True) or {}

    try:
        return flask.jsonify(
            model_manager.delete_model(
                payload.get("type"),
                payload.get("name"),
                current_model_path=current_config.get("model_path"),
                pipeline_running=is_pipeline_running(),
            )
        )
    except ModelValidationError as exc:
        return flask.jsonify({"error": str(exc)}), 400
    except ModelNotFoundError as exc:
        return flask.jsonify({"error": str(exc)}), 404
    except ModelConflictError as exc:
        return flask.jsonify({"error": str(exc)}), 409
    except ModelManagerError as exc:
        return flask.jsonify({"error": str(exc)}), 500


@app.route("/api/model-compile-jobs")
@auth.require_permission("models:view")
def api_model_compile_jobs():
    """
    List compile jobs for UI restore after close/refresh.
    ---
    tags:
      - Configuration
    responses:
      200:
        description: Compile job list
        schema:
          type: object
          properties:
            jobs:
              type: array
              items:
                type: object
                properties:
                  id:
                    type: string
                  status:
                    type: string
                  created_at:
                    type: string
                  started_at:
                    type: string
                  finished_at:
                    type: string
                  returncode:
                    type: integer
                  model:
                    type: object
                    properties:
                      type:
                        type: string
                      name:
                        type: string
                      source:
                        type: string
    """
    return flask.jsonify({"jobs": model_manager.list_compile_jobs()})


@app.route("/api/model-compile/<job_id>")
@auth.require_permission("models:view")
def api_model_compile_status(job_id):
    """
    Get compile job status and logs.
    ---
    tags:
      - Configuration
    parameters:
      - name: job_id
        in: path
        type: string
        required: true
        description: Compile job identifier returned by `/api/model-compile`.
    responses:
      200:
        description: Compile job status
        schema:
          type: object
          properties:
            id:
              type: string
            status:
              type: string
            created_at:
              type: string
            started_at:
              type: string
            finished_at:
              type: string
            returncode:
              type: integer
            command:
              type: array
              items:
                type: string
            model:
              type: object
            logs:
              type: array
              items:
                type: string
      404:
        description: Job not found.
    """
    try:
        return flask.jsonify(model_manager.get_compile_job(job_id))
    except ModelNotFoundError as exc:
        return flask.jsonify({"error": str(exc)}), 404


@app.route("/api/cameras")
@auth.require_permission("cameras:view")
def api_cameras():
    """
    List attached cameras and their modes.
    ---
    tags:
      - Configuration
    responses:
      200:
        description: List of V4L2 devices and supported modes
        schema:
          type: object
          properties:
            cameras:
              type: array
              items:
                type: object
                properties:
                  device:
                    type: string
                  name:
                    type: string
                  modes:
                    type: array
                    items:
                      type: object
                      properties:
                        width:
                          type: integer
                        height:
                          type: integer
                        fps:
                          type: integer
                        format:
                          type: string
      500:
        description: Camera enumeration failed.
    """
    try:
        cams = CameraManager.get_available_cameras()
        return flask.jsonify({"cameras": cams})
    except Exception as e:
        LOGGER.error(f"Failed to list cameras: {e}")
        return flask.jsonify({"error": str(e)}), 500


def check_port(host, port, timeout=0.1):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


@app.route("/api/status")
@auth.require_permission("status:view")
def api_status():
    """
    Get pipeline status.
    ---
    tags:
      - Control
    responses:
      200:
        description: Current state, threads, and MediaMTX status
        schema:
          type: object
          properties:
            state:
              type: string
              enum: ['idle', 'running']
            pipeline_id:
              type: string
            runtime:
              type: object
              properties:
                grid_count_enabled:
                  type: boolean
                  description: True when the grid feature is enabled.
                grid_score:
                  type: number
                  description: Current EMA grid quality score in the 0..1 range, or null before the first processed frame.
                grid_score_threshold:
                  type: number
                  description: EMA score threshold that auto-disables the grid feature. 0 disables auto-off.
                grid_auto_disabled:
                  type: boolean
                  description: True when the grid feature was auto-disabled by the score threshold.
            mediamtx:
              type: object
              properties:
                whep:
                  type: boolean
                rtsp:
                  type: boolean
            config:
              type: object
              properties:
                video_reference:
                  type: string
                model:
                  type: string
                model_task:
                  type: string
            threads:
              type: array
              items:
                type: string
            last_error:
              type: string
    """
    global pipeline_id
    thread_states = pipeline_thread_states()
    live_threads = [name for name, alive in thread_states if alive]
    state = "running" if any(alive for _, alive in thread_states) else "idle"
    
    video_desc = current_config.get("video_description", current_config.get("video", ""))
    pid_value = pipeline_id if pipeline and is_pipeline_running() else "-"
    current_model = current_config.get("model_path", "")

    mtx_whep = check_port("127.0.0.1", 8889)
    mtx_rtsp = check_port("127.0.0.1", 8554)

    return flask.jsonify({
        "state": state,
        "pipeline_id": pid_value,
        "last_error": last_error,
        "mediamtx": { "whep": mtx_whep, "rtsp": mtx_rtsp }, 
        "runtime": get_runtime_options(),
        "config": {
            "video_reference": video_desc,
            "model": os.path.basename(current_model) if current_model else "",
            "model_task": current_config.get("model_task"),
        },
        "threads": live_threads,
    })


@app.route("/api/results")
@auth.require_permission("results:view")
def api_list_results():
    """
    List all results.
    ---
    tags:
      - Results
    parameters:
      - name: date
        in: query
        type: string
        required: false
        description: Optional exact date in YYYY-MM-DD format.
    responses:
      200:
        description: List of processed videos
        schema:
          type: object
          properties:
            results:
              type: array
              items:
                type: object
                properties:
                  id:
                    type: string
                  timestamp:
                    type: string
                  duration:
                    type: string
                  duration_seconds:
                    type: number
                  detected_objects:
                    type: integer
                  class_counts:
                    type: object
                  s_value:
                    type: number
                  files:
                    type: array
                    items:
                      type: object
                      properties:
                        name:
                          type: string
                        path:
                          type: string
                        size:
                          type: string
                  owner_username:
                    type: string
      400:
        description: Invalid date filter.
      500:
        description: Result listing failed.
    """
    try:
        selected_date = _parse_results_date_filter()
        results_list = auth.filter_results_for_user(
            flask.g.current_user,
            HQ_OUTPUT_DIR,
            _collect_results_metadata(selected_date=selected_date),
        )
        for item in results_list:
            item.pop("_mtime", None)
            item.pop("run_path", None)
            item.pop("csv_path", None)
            item.pop("video_path", None)
            if auth.is_admin(flask.g.current_user):
                item["owner_username"] = auth.get_result_owner_username(HQ_OUTPUT_DIR, item["id"])

        return flask.jsonify({"results": results_list})

    except ValueError as exc:
        return flask.jsonify({"error": str(exc)}), 400
    except Exception as e:
        LOGGER.error(f"Failed to list results: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/api/results/search")
@auth.require_permission("results:view")
def api_search_results():
    """
    Search results by Analysis ID.
    ---
    tags:
      - Results
    parameters:
      - name: analysis_id
        in: query
        type: string
        required: false
        description: The Analysis ID or Timestamp to filter by
      - name: date
        in: query
        type: string
        required: false
        description: Optional exact date in YYYY-MM-DD format.
    responses:
      200:
        description: List of matching results with download links
        schema:
          type: object
          properties:
            results:
              type: array
              items:
                type: object
                properties:
                  analysis_id:
                    type: string
                  video_url:
                    type: string
                  csv_url:
                    type: string
                  images:
                    type: array
                    items:
                      type: object
                      properties:
                        name:
                          type: string
                        size:
                          type: string
                        url:
                          type: string
    """
    try:
        analysis_id = flask.request.args.get('analysis_id', '').strip()
        selected_date = _parse_results_date_filter()
        results_list = []
        host_url = flask.request.host_url.rstrip('/')

        visible_results = auth.filter_results_for_user(
            flask.g.current_user,
            HQ_OUTPUT_DIR,
            _collect_results_metadata(
                analysis_prefix=analysis_id,
                selected_date=selected_date,
            ),
        )

        for item in visible_results:
            if not item.get("video_path") and not item.get("csv_path"):
                continue

            run_id = item["id"]
            video_filename = f"{run_id}.mkv"
            csv_filename = f"{run_id}.csv"
            video_url = (
                _build_download_url(host_url, f"{run_id}/{video_filename}")
                if item.get("video_path")
                else None
            )
            csv_url = (
                _build_download_url(host_url, f"{run_id}/{csv_filename}")
                if item.get("csv_path")
                else None
            )
            size_mb = (
                round(os.path.getsize(item["video_path"]) / (1024 * 1024), 2)
                if item.get("video_path")
                else None
            )
            images = [
                {
                    "name": file_info["name"],
                    "size": file_info.get("size"),
                    "url": _build_download_url(host_url, file_info["path"]),
                }
                for file_info in item.get("files", [])
                if _is_result_image(file_info.get("name", ""))
            ]

            results_list.append({
                "analysis_id": run_id,
                "timestamp": item["timestamp"],
                "video_size": f"{size_mb} MB" if size_mb is not None else None,
                "video_url": video_url,
                "csv_url": csv_url,
                "images": images,
            })

        return flask.jsonify({"results": results_list})

    except ValueError as exc:
        return flask.jsonify({"error": str(exc)}), 400
    except Exception as e:
        LOGGER.error(f"Failed to search results: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/api/results/<pid>/last-row")
@auth.require_permission("results:inspect")
def api_result_last_csv_row(pid):
    """
    Return the last non-empty row from the result CSV as JSON.
    ---
    tags:
      - Results
    parameters:
      - name: pid
        in: path
        type: string
        required: true
        description: The Analysis ID to inspect
    responses:
      200:
        description: Last CSV row
        schema:
          type: object
          properties:
            analysis_id:
              type: string
            row:
              type: object
              properties:
                frame:
                  type: integer
                analysis_number:
                  type: string
                s_value:
                  type: number
                tilletia_objects:
                  type: integer
                class_counts:
                  type: object
                detections:
                  type: array
                  items:
                    type: object
                    properties:
                      bbox:
                        type: array
                        items:
                          type: integer
                      class_id:
                        type: integer
                      class_name:
                        type: string
                      confidence:
                        type: number
      400:
        description: Invalid analysis ID
      403:
        description: Authenticated user cannot inspect this result.
      404:
        description: CSV file or row not found
      500:
        description: Failed to read the CSV row.
    """
    try:
        if not pid or ".." in pid or "/" in pid:
            return flask.jsonify({"error": "Invalid ID"}), 400
        if not auth.user_can_access_result(flask.g.current_user, HQ_OUTPUT_DIR, pid):
            return flask.jsonify({"error": "Forbidden"}), 403

        csv_path = os.path.join(HQ_OUTPUT_DIR, pid, f"{pid}.csv")
        if not os.path.isfile(csv_path):
            return flask.jsonify({"error": "CSV not found"}), 404

        row = _read_last_csv_row(csv_path)
        if row is None:
            return flask.jsonify({"error": "CSV is empty"}), 404

        return flask.jsonify({
            "analysis_id": pid,
            "row": _coerce_csv_row(row),
        })

    except Exception as exc:
        LOGGER.error("Failed to read last CSV row for %s: %s", pid, exc)
        return flask.jsonify({"error": str(exc)}), 500


@app.route("/api/results/<pid>", methods=["DELETE"])
@auth.require_permission("results:delete")
def api_delete_result(pid):
    """
    Delete a result set.
    ---
    tags:
      - Results
    parameters:
      - name: pid
        in: path
        type: string
        required: true
        description: The Analysis ID to delete
    responses:
      200:
        description: Files deleted successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            deleted:
              type: array
              items:
                type: string
      400:
        description: Invalid analysis ID.
      500:
        description: Result deletion failed.
    """
    try:
        if not pid or ".." in pid or "/" in pid:
            return flask.jsonify({"error": "Invalid ID"}), 400

        run_dir = os.path.join(HQ_OUTPUT_DIR, pid)
        deleted_files = []

        if os.path.isdir(run_dir):
            for root, _, files in os.walk(run_dir):
                for name in files:
                    rel_path = os.path.relpath(os.path.join(root, name), HQ_OUTPUT_DIR)
                    deleted_files.append(rel_path)
            shutil.rmtree(run_dir)
        auth.delete_result_owner(pid)

        LOGGER.info(f"Deleted results for ID {pid}: {deleted_files}")
        return flask.jsonify({"success": True, "deleted": deleted_files})

    except Exception as e:
        LOGGER.error(f"Failed to delete results for {pid}: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/api/results/<pid>/process-source", methods=["POST"])
@auth.require_permission("dashboard:configure")
def api_prepare_result_process_source(pid):
    """
    Reuse a processed result video as the current dashboard file source.
    ---
    tags:
      - Results
    responses:
      200:
        description: Result video prepared as the dashboard file source.
        schema:
          type: object
          properties:
            success:
              type: boolean
            video:
              type: string
            file_name:
              type: string
      403:
        description: Authenticated user cannot reuse this result.
      404:
        description: Result or result video not found.
      500:
        description: Failed to prepare the result video for processing.
    """
    try:
        if not auth.user_can_access_result(flask.g.current_user, HQ_OUTPUT_DIR, pid):
            return flask.jsonify({"error": "Forbidden"}), 403

        if _build_result_metadata(pid) is None:
            return flask.jsonify({"error": "Result not found."}), 404

        payload = flask.request.get_json(silent=True) or {}
        try:
            video_path = _resolve_result_video_path(pid, payload.get("path", ""))
        except ValueError as exc:
            return flask.jsonify({"error": str(exc)}), 400
        except FileNotFoundError as exc:
            return flask.jsonify({"error": str(exc)}), 404

        auth.update_dashboard_settings(
            {
                "source_type": "file",
                "uploaded_path": video_path,
            }
        )

        return flask.jsonify(
            {
                "success": True,
                "video": video_path,
                "file_name": os.path.basename(video_path),
            }
        )
    except Exception as e:
        LOGGER.error(f"Failed to prepare result video for processing {pid}: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/download/<path:filename>")
@auth.require_permission("results:download")
def download_file(filename):
    """
    Download a file from the HQ output directory.
    ---
    tags:
      - Results
    parameters:
      - name: filename
        in: path
        type: string
        required: true
        description: Relative file path within the results directory
    responses:
      200:
        description: File download
      403:
        description: Authenticated user cannot access this result file.
      404:
        description: Not found
    """
    try:
        run_id = (filename or "").split("/", 1)[0]
        if not run_id or not auth.user_can_access_result(flask.g.current_user, HQ_OUTPUT_DIR, run_id):
            return flask.jsonify({"error": "Forbidden"}), 403
        return flask.send_from_directory(HQ_OUTPUT_DIR, filename, as_attachment=True)
    except Exception as e:
        return flask.jsonify({"error": str(e)}), 404


@app.route("/vendor/<path:filename>")
def vendor_file(filename):
    """
    Serve bundled static frontend assets.
    ---
    tags:
      - Static
    parameters:
      - name: filename
        in: path
        type: string
        required: true
        description: Relative asset path within the bundled vendor directory.
    responses:
      200:
        description: Static asset response.
      404:
        description: Asset not found.
    """
    try:
        return flask.send_from_directory(VENDOR_DIR, filename)
    except Exception as e:
        return flask.jsonify({"error": str(e)}), 404


@app.route("/api/results/<pid>/download")
@auth.require_permission("results:download")
def api_download_result(pid):
    """
    Download a result folder as a ZIP archive.
    ---
    tags:
      - Results
    parameters:
      - name: pid
        in: path
        type: string
        required: true
        description: The Analysis ID to download
    responses:
      200:
        description: ZIP archive of the result folder
      400:
        description: Invalid analysis ID.
      403:
        description: Authenticated user cannot access this result.
      404:
        description: Not found
    """
    try:
        if not pid or ".." in pid or "/" in pid:
            return flask.jsonify({"error": "Invalid ID"}), 400
        if not auth.user_can_access_result(flask.g.current_user, HQ_OUTPUT_DIR, pid):
            return flask.jsonify({"error": "Forbidden"}), 403

        run_dir = os.path.join(HQ_OUTPUT_DIR, pid)
        if not os.path.isdir(run_dir):
            return flask.jsonify({"error": "Not found"}), 404

        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(run_dir):
                for name in files:
                    full_path = os.path.join(root, name)
                    rel_path = os.path.relpath(full_path, run_dir)
                    zf.write(full_path, rel_path)

        archive.seek(0)
        return flask.send_file(
            archive,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{pid}.zip",
        )
    except Exception as e:
        LOGGER.error(f"Failed to create zip for {pid}: {e}")
        return flask.jsonify({"error": str(e)}), 500


@app.route("/api/snapshot", methods=["POST"])
@auth.require_permission("dashboard:view")
def api_snapshot():
    """
    Capture a manual snapshot from the active pipeline.
    ---
    tags:
      - Control
    responses:
      200:
        description: Snapshot saved successfully.
        schema:
          type: object
          properties:
            success:
              type: boolean
            pipeline_id:
              type: string
            filename:
              type: string
            path:
              type: string
      400:
        description: No pipeline is currently running.
      504:
        description: Timed out waiting for the next frame.
    """
    global pipeline, pipeline_id

    if pipeline is None or not is_pipeline_running():
        return flask.jsonify({"error": "Pipeline is not running."}), 400

    try:
        snapshot = pipeline.request_snapshot(timeout=5.0)
        return flask.jsonify(
            {
                "success": True,
                "pipeline_id": pipeline_id,
                "filename": snapshot["filename"],
                "path": snapshot["path"],
            }
        )
    except TimeoutError as exc:
        return flask.jsonify({"error": str(exc)}), 504
    except Exception as exc:
        LOGGER.exception("Failed to capture manual snapshot: %s", exc)
        return flask.jsonify({"error": str(exc)}), 500


@app.route("/api/start", methods=["POST"])
@auth.require_permission("pipeline:start")
def api_start():
    """
    Start the inference pipeline.
    ---
    tags:
      - Control
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            analysis_number:
              type: string
              description: Optional custom ID
            source_type:
              type: string
              enum: ['camera', 'file']
            device:
              type: string
              description: Device path (e.g., /dev/video0)
            width:
              type: integer
            height:
              type: integer
            fps:
              type: integer
            format:
              type: string
              description: Camera pixel format, for example `MJPG`.
            video:
              type: string
              description: Path to uploaded video file
            model_path:
              type: string
            model_task:
              type: string
              enum: ['segment', 'detect', 'auto']
              description: Optional Ultralytics task override. When omitted, the saved catalog setting is used and defaults to segment.
            vis_conf:
              type: number
            vis_strategy:
              type: string
            grid_count_enabled:
              type: boolean
              description: Optional initial grid feature state. When enabled, grid detection runs and counting is limited to the detected viewport.
            grid_debug_enabled:
              type: boolean
              description: Optional initial grid debug overlay state.
            grid_score_threshold:
              type: number
              minimum: 0
              maximum: 1
              description: Optional EMA grid score threshold used to auto-disable the grid feature. Use 0 to disable auto-off.
    responses:
      200:
        description: Pipeline started successfully
        schema:
          type: object
          properties:
            success:
              type: boolean
            pipeline_id:
              type: string
      400:
        description: Already running or invalid input
      403:
        description: Authenticated user cannot override admin-controlled dashboard settings.
      500:
        description: Pipeline start failed.
    """
    global pipeline, last_error, current_config, pipeline_id

    if is_pipeline_running():
        return flask.jsonify({"error": "already running"}), 400

    if pipeline is not None and not is_pipeline_running():
        try:
            pipeline.__exit__(None, None, None)
        except Exception:
            LOGGER.exception("Error cleaning up stale pipeline")
        finally:
            pipeline = None

    last_error = None
    tmp_pipeline = None

    try:
        raw_data = flask.request.get_json(silent=True) or {}
        try:
            data = auth.resolve_dashboard_start_payload(raw_data, flask.g.current_user)
        except PermissionError as exc:
            return flask.jsonify({"error": str(exc)}), 403
        
        source_type = data.get("source_type", "file")
        set_grid_score(None)
        set_grid_count_enabled(
            data.get("grid_count_enabled", get_grid_count_enabled())
        )
        set_grid_debug_enabled(
            data.get("grid_debug_enabled", get_grid_debug_enabled())
        )
        set_grid_score_threshold(
            data.get("grid_score_threshold", get_grid_score_threshold())
        )
        
        args_dict = dict()
        if os.path.exists(CONFIG_FILEPATH):
            with open(CONFIG_FILEPATH, 'r') as config_input:
                args_dict = json.load(config_input)

        analysis_num = data.get("analysis_number", "").strip()
        if analysis_num:
            pipeline_id = generate_unique_result_run_id(analysis_num, HQ_OUTPUT_DIR)
        else:
            start_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            pipeline_id = f"{start_time_str}-{uuid.uuid4().hex}"
        
        model_path = data.get("model_path", None)
        
        if not model_path:
            model_path = "/app/model/weights-fp16.engine"

        requested_model_task = model_manager.sanitize_ul_model_task(data.get("model_task"))
        resolved_model_task = requested_model_task or model_manager.resolve_model_task_for_path(model_path)
        runtime_model_settings = model_manager.resolve_model_runtime_settings_for_path(model_path)

        if "vis_conf" in raw_data:
            requested_conf = float(raw_data.get("vis_conf"))
        else:
            requested_conf = model_manager.resolve_model_default_confidence_threshold_for_path(
                model_path,
                fallback=data.get("vis_conf", model_manager.DEFAULT_MODEL_CONFIDENCE_THRESHOLD),
            )
        vis_strategy = data.get("vis_strategy", "tracker")
        
        args_dict.update(dict(
            print_every=60,
            stream_host="127.0.0.1",
            stream_port=8554,
            log_level="INFO",
            output_path="pub-output",
            model_conf=0.10,
            vis_conf=requested_conf,
            vis_strategy=vis_strategy,
            pipeline_id=pipeline_id,
            hq_output_dir=HQ_OUTPUT_DIR,
            output_stream='WebRTC',
            model_path=model_path,
            model_task=resolved_model_task,
            model_inference_width=runtime_model_settings.get("inference_width"),
            model_inference_height=runtime_model_settings.get("inference_height"),
        ))

        if source_type == "camera":
            device = data.get("device")
            width = int(data.get("width", 1280))
            height = int(data.get("height", 720))
            fps = int(data.get("fps", 30))
            pixel_format = data.get("format", "MJPG")
            
            if not device:
                default_camera, default_mode = CameraManager.get_default_camera_selection()
                if default_camera is None or default_mode is None:
                    raise ValueError("No device selected")
                device = default_camera["device"]
                width = int(default_mode["width"])
                height = int(default_mode["height"])
                fps = int(default_mode["fps"])
                pixel_format = default_mode["format"]

            args_dict["mode"] = "v4l2-gs"
            args_dict["device"] = device
            args_dict["width"] = width
            args_dict["height"] = height
            args_dict["fps"] = fps
            args_dict["pixel_format"] = pixel_format
            
            video_desc = f"{device} ({width}x{height} @ {fps}fps {pixel_format})"
            
            args = argparse.Namespace(**args_dict)
            reader = V4L2StreamReader(args.device, args.width, args.height, args.fps, pixel_format=pixel_format)

        else:
            video = data.get("video")
            if not video:
                raise ValueError("No file provided")

            video_path = _resolve_video_input_path(video)
            if not os.path.isfile(video_path):
                raise ValueError("File not found: %s (resolved: %s)" % (video, video_path))
                
            args_dict["mode"] = "file"
            video_desc = os.path.basename(video_path)
            args_dict["fps"] = 30 
            
            args = argparse.Namespace(**args_dict)
            reader = FileReader(video_path, args.fps)
            
            with reader:
                cap = reader.cap
                if cap is None or not cap.isOpened():
                    raise RuntimeError("Failed to open file")
                
                args.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                args.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                f_fps = cap.get(cv2.CAP_PROP_FPS)
                if f_fps > 0:
                    args.fps = int(f_fps)
                LOGGER.info(f"File probed: {args.width}x{args.height} @ {args.fps}")

        args.grid_count_enabled = get_grid_count_enabled()
        args.grid_count_enabled_getter = get_grid_count_enabled
        args.grid_count_enabled_setter = set_grid_count_enabled
        args.grid_debug_enabled = get_grid_debug_enabled()
        args.grid_debug_enabled_getter = get_grid_debug_enabled
        args.grid_score_setter = set_grid_score
        args.grid_score_threshold = get_grid_score_threshold()
        args.grid_score_threshold_getter = get_grid_score_threshold

        LOGGER.info(f"Starting pipeline: {args}")
        tmp_pipeline = StreamPipeline(reader, args)
        tmp_pipeline.__enter__()
        pipeline = tmp_pipeline

        current_config = {
            "video_description": video_desc,
            "model_path": model_path,
            "model_task": resolved_model_task,
            "model_inference_width": runtime_model_settings.get("inference_width"),
            "model_inference_height": runtime_model_settings.get("inference_height"),
        }
        auth.store_result_owner(pipeline_id, flask.g.current_user["id"])

        if auth.user_has_permission(flask.g.current_user, "dashboard:configure"):
            auth.update_dashboard_settings(
                {
                    "analysis_number": data.get("analysis_number", ""),
                    "source_type": source_type,
                    "camera_device": data.get("device", ""),
                    "camera_mode": {
                        "width": data.get("width", 1280),
                        "height": data.get("height", 720),
                        "fps": data.get("fps", 30),
                        "format": data.get("format", "MJPG"),
                    },
                    "uploaded_path": data.get("video", ""),
                    "model_path": model_path,
                    "vis_conf": requested_conf,
                    "grid_count_enabled": data.get("grid_count_enabled", get_grid_count_enabled()),
                    "grid_debug_enabled": data.get("grid_debug_enabled", get_grid_debug_enabled()),
                    "grid_score_threshold": data.get("grid_score_threshold", get_grid_score_threshold()),
                }
            )

        return flask.jsonify({"success": True, "pipeline_id": pipeline_id})

    except Exception as e:
        last_error = str(e)
        LOGGER.exception("Failed to start pipeline: %s", e)
        if tmp_pipeline is not None:
            try:
                tmp_pipeline.__exit__(type(e), e, e.__traceback__)
            except Exception:
                pass
        return flask.jsonify({"error": str(e)}), 500


@app.route("/api/stop", methods=["POST"])
@auth.require_permission("pipeline:stop")
def api_stop():
    """
    Stop the inference pipeline.
    ---
    tags:
      - Control
    responses:
      200:
        description: Pipeline stopped
        schema:
          type: object
          properties:
            success:
              type: boolean
      500:
        description: Pipeline stop failed.
    """
    global pipeline, last_error, current_config, pipeline_id

    if pipeline is None and not is_pipeline_running():
        current_config = {}
        return flask.jsonify({"success": True})

    try:
        if pipeline is not None:
            pipeline.__exit__(None, None, None)

        pipeline, pipeline_id = None, None
        current_config = {}

        return flask.jsonify({"success": True})
    except Exception as e:
        last_error = str(e)
        LOGGER.exception("Failed to stop pipeline: %s", e)
        return flask.jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
@auth.require_permission("upload:video")
def api_upload():
    """
    Upload a video file.
    ---
    tags:
      - Configuration
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: The video file to upload
    responses:
      200:
        description: File uploaded successfully
        schema:
          type: object
          properties:
            video:
              type: string
              description: Server path to the uploaded file
      400:
        description: Missing file or invalid filename.
      500:
        description: Upload failed.
    """
    if "file" not in flask.request.files:
        return flask.jsonify({"error": "no file"}), 400

    file = flask.request.files["file"]
    if file.filename == "":
        return flask.jsonify({"error": "no filename"}), 400

    filename = os.path.basename(file.filename)
    if not filename:
        return flask.jsonify({"error": "invalid filename"}), 400

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    path = os.path.join(UPLOAD_DIR, filename)
    file.save(path)
    try:
        auth.update_dashboard_settings({"uploaded_path": path})
    except (TypeError, ValueError):
        pass

    return flask.jsonify({"video": path})


@app.route("/<path:path>/whep", methods=["GET", "POST", "OPTIONS"])
@auth.require_permission("dashboard:view")
def proxy_whep(path):
    """
    Proxy WHEP signaling requests to the local MediaMTX instance.
    ---
    tags:
      - Streaming
    parameters:
      - name: path
        in: path
        type: string
        required: true
        description: Stream path forwarded to the upstream `/<path>/whep` endpoint.
    responses:
      200:
        description: Successful proxied WHEP response.
      201:
        description: Successful proxied WHEP response created by the upstream service.
      204:
        description: Successful proxied WHEP response without content.
      401:
        description: Missing or invalid access token.
      403:
        description: Authenticated user lacks permission to view the dashboard stream.
    """
    # Proxy WHEP requests to the local mediamtx server
    target_url = "http://127.0.0.1:8889/%s/whep" % path
    headers = {k: v for k, v in flask.request.headers.items() if k.lower() != "host"}

    if flask.request.method == "OPTIONS":
        resp = requests.options(target_url, headers=headers)
    elif flask.request.method == "POST":
        resp = requests.post(target_url, data=flask.request.data, headers=headers)
    else:
        resp = requests.get(target_url, headers=headers)

    excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    response_headers = [(name, value) for (name, value) in resp.raw.headers.items() if name.lower() not in excluded_headers]
    return flask.Response(resp.content, resp.status_code, response_headers)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=8000, threaded=True)
