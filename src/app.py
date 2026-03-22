import argparse
import csv
import json
import logging
import os
import uuid
import glob
import shutil
import io
import zipfile
import socket
import re
import sys
import threading
import subprocess
import time
from datetime import datetime
from collections import deque
from urllib.parse import quote

import cv2
import requests
from flask import Flask, Response, jsonify, request, send_file, send_from_directory
from flasgger import Swagger

from inference_pipeline import StreamPipeline
from stream_readers import V4L2StreamReader, FileReader
from camera_manager import CameraManager

LOGGER = logging.getLogger("app")
CONFIG_FILEPATH = "/app/config.json"
HQ_OUTPUT_DIR = "/app/output_hq"
MODEL_DIR = "/app/model"
MODEL_TASKS_FILEPATH = os.path.join(MODEL_DIR, "model_tasks.json")
VENDOR_DIR = "/opt/web/vendor"
VALID_UL_MODEL_TASKS = ("segment", "detect", "auto")
MODEL_UPLOAD_EXTENSIONS = (".pt",)
MODEL_COMPILE_METADATA_SUFFIX = ".compile.json"
RESULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

app = Flask(__name__)

# --- SWAGGER CONFIGURATION ---
app.config['SWAGGER'] = {
    'title': 'Desktop Inference Pipeline API',
    'uiversion': 3,
    'specs_route': '/api/docs/'  # The URL where Swagger UI will be available
}

# Initialize Flasgger
swagger = Swagger(app)

# Global runtime state
pipeline = None  # type: StreamPipeline | None
current_config = {}
last_error = None  # type: str | None
pipeline_id = None  # type: str | None
compile_jobs = {}
compile_jobs_lock = threading.Lock()
model_task_overrides_lock = threading.Lock()
runtime_options = {
    "grid_count_enabled": True,
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
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _coerce_threshold(value):
    if value is None or value == "":
        return 0.0
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        return 0.0
    if threshold < 0.0:
        return 0.0
    if threshold > 1.0:
        return 1.0
    return threshold


def get_grid_count_enabled():
    with runtime_options_lock:
        return bool(runtime_options.get("grid_count_enabled", True))


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
            "grid_count_enabled": bool(runtime_options.get("grid_count_enabled", True)),
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


def _append_compile_log(job_id, line):
    with compile_jobs_lock:
        job = compile_jobs.get(job_id)
        if job is None:
            return
        job["logs"].append(line)


def _find_active_job_for_model(model_type, model_name):
    active_states = {"queued", "running"}
    selected = None
    for job in compile_jobs.values():
        model = job.get("model") or {}
        if model.get("type") != model_type or model.get("name") != model_name:
            continue
        if job.get("status") not in active_states:
            continue
        created = job.get("created_at", "")
        if selected is None or created > selected.get("created_at", ""):
            selected = job
    return selected


def _model_task_key(model_type, model_name):
    return f"{model_type}:{model_name}"


def _sanitize_ul_model_task(task):
    if not isinstance(task, str):
        return None
    normalized = task.strip().lower()
    if normalized in VALID_UL_MODEL_TASKS:
        return normalized
    return None


def _read_model_task_overrides_unlocked():
    if not os.path.exists(MODEL_TASKS_FILEPATH):
        return {}

    try:
        with open(MODEL_TASKS_FILEPATH, "r") as task_input:
            data = json.load(task_input)
    except FileNotFoundError:
        return {}
    except Exception as exc:
        LOGGER.error("Failed to read model task config %s: %s", MODEL_TASKS_FILEPATH, exc)
        return {}

    if not isinstance(data, dict):
        return {}

    overrides = {}
    for key, value in data.items():
        if not isinstance(key, str):
            continue
        task = _sanitize_ul_model_task(value)
        if task is not None:
            overrides[key] = task
    return overrides


def _write_model_task_overrides_unlocked(overrides):
    sanitized = {}
    for key, value in (overrides or {}).items():
        if not isinstance(key, str):
            continue
        task = _sanitize_ul_model_task(value)
        if task is not None:
            sanitized[key] = task

    os.makedirs(os.path.dirname(MODEL_TASKS_FILEPATH), exist_ok=True)
    tmp_path = f"{MODEL_TASKS_FILEPATH}.tmp"
    with open(tmp_path, "w") as task_output:
        json.dump(sanitized, task_output, indent=2, sort_keys=True)
        task_output.write("\n")
    os.replace(tmp_path, MODEL_TASKS_FILEPATH)


def get_catalog_model_task(model_type, model_name):
    if model_type != "ul":
        return None

    with model_task_overrides_lock:
        overrides = _read_model_task_overrides_unlocked()
    return overrides.get(_model_task_key(model_type, model_name), "segment")


def set_catalog_model_task(model_type, model_name, task):
    if model_type != "ul":
        raise ValueError("Task override is only supported for Ultralytics models.")

    normalized_task = _sanitize_ul_model_task(task)
    if normalized_task is None:
        raise ValueError("Invalid model task.")

    with model_task_overrides_lock:
        overrides = _read_model_task_overrides_unlocked()
        key = _model_task_key(model_type, model_name)
        if normalized_task == "segment":
            overrides.pop(key, None)
        else:
            overrides[key] = normalized_task
        _write_model_task_overrides_unlocked(overrides)

    return normalized_task


def clear_catalog_model_task(model_type, model_name):
    with model_task_overrides_lock:
        overrides = _read_model_task_overrides_unlocked()
        changed = False
        for candidate in {model_name, model_name.removesuffix("-fp16")}:
            key = _model_task_key(model_type, candidate)
            if key in overrides:
                overrides.pop(key, None)
                changed = True
        if changed:
            _write_model_task_overrides_unlocked(overrides)


def resolve_model_task_for_path(model_path):
    if not model_path:
        return "segment"

    normalized = os.path.normpath(model_path)
    try:
        relative = os.path.relpath(normalized, MODEL_DIR)
    except ValueError:
        return "segment"

    parts = relative.split(os.sep)
    if len(parts) < 2:
        return "segment"

    model_type = parts[0]
    if model_type != "ul":
        return "auto"

    base_name = os.path.splitext(parts[-1])[0]
    candidates = [base_name]
    if base_name.endswith("-fp16"):
        candidates.append(base_name[:-5])

    with model_task_overrides_lock:
        overrides = _read_model_task_overrides_unlocked()

    for candidate in candidates:
        task = overrides.get(_model_task_key(model_type, candidate))
        if task is not None:
            return task
    return "segment"


def get_current_tensorrt_version():
    try:
        import tensorrt as trt
        return str(trt.__version__)
    except Exception:
        return None


def _compile_metadata_path(engine_path):
    if not engine_path:
        return None
    return f"{engine_path}{MODEL_COMPILE_METADATA_SUFFIX}"


def _read_compile_metadata(engine_path):
    metadata_path = _compile_metadata_path(engine_path)
    if not metadata_path or not os.path.isfile(metadata_path):
        return {}
    try:
        with open(metadata_path, "r", encoding="utf-8") as fp:
            data = json.load(fp)
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        LOGGER.warning("Failed to read compile metadata %s: %s", metadata_path, exc)
        return {}


def _write_compile_metadata(engine_path, *, model_type=None, source_path=None, command=None):
    metadata_path = _compile_metadata_path(engine_path)
    if not metadata_path:
        return
    payload = {
        "tensorrt_version": get_current_tensorrt_version(),
        "compiled_at": datetime.utcnow().isoformat() + "Z",
    }
    if model_type:
        payload["model_type"] = model_type
    if source_path:
        payload["source_path"] = source_path
    if command:
        payload["command"] = list(command)
    try:
        with open(metadata_path, "w", encoding="utf-8") as fp:
            json.dump(payload, fp, indent=2, sort_keys=True)
    except Exception as exc:
        LOGGER.warning("Failed to write compile metadata %s: %s", metadata_path, exc)


def _resolve_compiled_engine_path(model_type, source_path):
    if not source_path:
        return None

    source_dir = os.path.dirname(source_path)
    source_base = os.path.splitext(os.path.basename(source_path))[0]
    candidates = []
    if model_type == "rf":
        candidates.extend([
            os.path.join(source_dir, f"{source_base}-fp16.engine"),
            os.path.join(source_dir, f"{source_base}.engine"),
        ])
    else:
        candidates.extend([
            os.path.join(source_dir, f"{source_base}.engine"),
            os.path.join(source_dir, f"{source_base}-fp16.engine"),
        ])

    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _record_compile_metadata(job_id):
    with compile_jobs_lock:
        job = compile_jobs.get(job_id)
        if job is None:
            return
        model = job.get("model", {})
        source_path = model.get("source")
        model_type = model.get("type")
        command = job.get("command")

    engine_path = _resolve_compiled_engine_path(model_type, source_path)
    if not engine_path:
        return

    _write_compile_metadata(
        engine_path,
        model_type=model_type,
        source_path=source_path,
        command=command,
    )
    _append_compile_log(job_id, f"Saved compile metadata: {_compile_metadata_path(engine_path)}")


def _cleanup_ul_compile_intermediates(job_id):
    with compile_jobs_lock:
        job = compile_jobs.get(job_id)
        if job is None:
            return
        model = job.get("model", {})
        source_path = model.get("source")
        model_type = model.get("type")
        started_ts = job.get("started_ts", time.time())

    if model_type != "ul" or not source_path:
        return

    source_dir = os.path.dirname(source_path)
    source_base = os.path.splitext(os.path.basename(source_path))[0]
    candidates = set()
    for pattern in (
        os.path.join(source_dir, f"{source_base}*.onnx"),
        os.path.join("/app/runs", "**", f"{source_base}*.onnx"),
    ):
        for path in glob.glob(pattern, recursive=True):
            if os.path.isfile(path):
                candidates.add(path)

    for path in sorted(candidates):
        try:
            if os.path.getmtime(path) + 2 < started_ts:
                continue
        except OSError:
            continue
        try:
            os.remove(path)
            _append_compile_log(job_id, f"Removed intermediate artifact: {path}")
        except Exception as exc:
            _append_compile_log(job_id, f"Failed to remove intermediate artifact {path}: {exc}")


def _sanitize_uploaded_model_filename(filename):
    safe_name = os.path.basename((filename or "").strip())
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", safe_name)
    return safe_name.strip("._")


def collect_model_artifact_paths(entry):
    if not entry:
        return []

    base_dir = os.path.join(MODEL_DIR, entry["type"])
    base_name = entry["name"]
    artifacts = set(entry.get("source_paths") or [])

    engine = entry.get("engine") or {}
    engine_path = engine.get("path")
    if engine_path:
        artifacts.add(engine_path)
        metadata_path = _compile_metadata_path(engine_path)
        if metadata_path:
            artifacts.add(metadata_path)

    for suffix in (".pt", ".onnx", ".engine"):
        artifacts.add(os.path.join(base_dir, f"{base_name}{suffix}"))
    if not base_name.endswith("-fp16"):
        artifacts.add(os.path.join(base_dir, f"{base_name}-fp16.engine"))

    for candidate in list(artifacts):
        if candidate.endswith(".engine"):
            metadata_path = _compile_metadata_path(candidate)
            if metadata_path:
                artifacts.add(metadata_path)

    return sorted(path for path in artifacts if os.path.isfile(path))


def _normalize_ul_engine_output(job_id):
    with compile_jobs_lock:
        job = compile_jobs.get(job_id)
        if job is None:
            return
        model = job.get("model", {})
        source_pt = model.get("source")
        started_ts = job.get("started_ts", time.time())

    if not source_pt:
        return

    source_dir = os.path.dirname(source_pt)
    source_base = os.path.splitext(os.path.basename(source_pt))[0]
    expected_engine = os.path.join(source_dir, f"{source_base}.engine")
    if os.path.exists(expected_engine):
        _append_compile_log(job_id, f"Engine ready: {expected_engine}")
        return

    candidates = []
    search_patterns = [
        os.path.join(source_dir, "*.engine"),
        "/app/*.engine",
        "/app/runs/**/*.engine",
    ]
    for pattern in search_patterns:
        for path in glob.glob(pattern, recursive=True):
            if not os.path.isfile(path):
                continue
            try:
                mtime = os.path.getmtime(path)
            except OSError:
                continue
            if mtime + 2 < started_ts:
                continue
            score = 0
            filename = os.path.basename(path)
            if source_base in filename:
                score += 10
            if path.startswith(source_dir):
                score += 5
            candidates.append((score, mtime, path))

    if not candidates:
        _append_compile_log(
            job_id,
            "Compile finished, but no engine artifact was found in expected paths.",
        )
        return

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    best_path = candidates[0][2]
    if best_path == expected_engine:
        _append_compile_log(job_id, f"Engine ready: {expected_engine}")
        return

    try:
        shutil.move(best_path, expected_engine)
        _append_compile_log(
            job_id,
            f"Moved engine artifact to model folder: {expected_engine}",
        )
    except Exception as exc:
        _append_compile_log(
            job_id,
            f"Failed to move engine artifact to {expected_engine}: {exc}",
        )


def _run_compile_job(job_id, command, cwd=None):
    with compile_jobs_lock:
        job = compile_jobs.get(job_id)
        if job is None:
            return
        job["status"] = "running"
        job["started_at"] = datetime.utcnow().isoformat() + "Z"
        job["started_ts"] = time.time()
        job["command"] = command

    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
    except Exception as exc:
        _append_compile_log(job_id, f"Failed to start compile: {exc}")
        with compile_jobs_lock:
            job = compile_jobs.get(job_id)
            if job is not None:
                job["status"] = "error"
                job["finished_at"] = datetime.utcnow().isoformat() + "Z"
                job["returncode"] = None
        return

    total_lines = 0
    last_heartbeat = time.time()
    try:
        for line in process.stdout:
            if line:
                total_lines += 1
                clean = line.rstrip()
                _append_compile_log(job_id, clean)

                now = time.time()
                if now - last_heartbeat >= 15:
                    with compile_jobs_lock:
                        job = compile_jobs.get(job_id)
                        started_ts = (job or {}).get("started_ts", now)
                    elapsed = int(now - started_ts)
                    _append_compile_log(
                        job_id,
                        f"[still running] {elapsed}s elapsed, processed {total_lines} output lines",
                    )
                    last_heartbeat = now
    except Exception as exc:
        _append_compile_log(job_id, f"Error reading output: {exc}")

    process.wait()
    with compile_jobs_lock:
        model_type = (compile_jobs.get(job_id) or {}).get("model", {}).get("type")
    if process.returncode == 0:
        if model_type == "ul":
            _normalize_ul_engine_output(job_id)
            _cleanup_ul_compile_intermediates(job_id)
        _record_compile_metadata(job_id)

    with compile_jobs_lock:
        job = compile_jobs.get(job_id)
        if job is not None:
            job["returncode"] = process.returncode
            job["finished_at"] = datetime.utcnow().isoformat() + "Z"
            job["status"] = "done" if process.returncode == 0 else "error"


def build_model_catalog():
    model_map = {}
    model_types = ["ul", "rf"]
    model_exts = [".onnx", ".pt"]

    for model_type in model_types:
        base_dir = os.path.join(MODEL_DIR, model_type)
        if not os.path.isdir(base_dir):
            continue

        for ext in model_exts:
            for path in glob.glob(os.path.join(base_dir, f"*{ext}")):
                filename = os.path.basename(path)
                base = os.path.splitext(filename)[0]
                key = (model_type, base)
                entry = model_map.get(key)
                if entry is None:
                    entry = {
                        "name": base,
                        "type": model_type,
                        "sources": [],
                        "source_paths": [],
                    }
                    model_map[key] = entry

                entry["sources"].append(ext.lstrip("."))
                entry["source_paths"].append(path)
        for path in glob.glob(os.path.join(base_dir, "*.engine")):
            filename = os.path.basename(path)
            base = os.path.splitext(filename)[0]
            key = (model_type, base)
            if key not in model_map:
                model_map[key] = {
                    "name": base,
                    "type": model_type,
                    "sources": [],
                    "source_paths": [],
                }

    models = []
    for (model_type, base), entry in sorted(model_map.items(), key=lambda item: item[0]):
        base_dir = os.path.join(MODEL_DIR, model_type)
        engine_name = f"{base}.engine"
        engine_path = os.path.join(base_dir, engine_name)
        engine = None

        if os.path.exists(engine_path):
            engine = {"name": engine_name, "path": engine_path}
        elif not base.endswith("-fp16"):
            fp16_name = f"{base}-fp16.engine"
            fp16_path = os.path.join(base_dir, fp16_name)
            if os.path.exists(fp16_path):
                engine = {"name": fp16_name, "path": fp16_path}

        if engine is not None:
            engine_metadata = _read_compile_metadata(engine["path"])
            if engine_metadata:
                engine["tensorrt_version"] = engine_metadata.get("tensorrt_version")
                engine["compiled_at"] = engine_metadata.get("compiled_at")

        entry["engine"] = engine
        entry["compiled"] = engine is not None
        entry["display"] = f"[{model_type.upper()}] {base}"
        entry["task"] = get_catalog_model_task(model_type, base)
        models.append(entry)

    return models


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
    return _parse_iso_date(request.args.get("date", "").strip(), "date")


def _result_matches_date(timestamp_value, selected_date=None):
    if not timestamp_value:
        return selected_date is None

    result_date = datetime.fromtimestamp(timestamp_value).date()
    return selected_date is None or result_date == selected_date


def _build_result_metadata(run_id):
    run_path = os.path.join(HQ_OUTPUT_DIR, run_id)
    if not os.path.isdir(run_path):
        return None

    files = []
    latest_mtime = 0
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

    timestamp = datetime.fromtimestamp(latest_mtime).strftime('%Y-%m-%d %H:%M:%S') if latest_mtime else "-"
    csv_path = os.path.join(run_path, f"{run_id}.csv")
    video_path = os.path.join(run_path, f"{run_id}.mkv")

    return {
        "id": run_id,
        "run_path": run_path,
        "timestamp": timestamp,
        "files": files,
        "_mtime": latest_mtime,
        "csv_path": csv_path if os.path.isfile(csv_path) else None,
        "video_path": video_path if os.path.isfile(video_path) else None,
    }


def _build_download_url(host_url, relative_path):
    return f"{host_url}/download/{quote(relative_path, safe='/')}"


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
        if analysis_prefix and not run_id.startswith(analysis_prefix):
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

    detections = []
    for chunk in value.split("|"):
        chunk = chunk.strip()
        if not chunk:
            continue

        parts = {}
        for item in chunk.split("_"):
            if "=" not in item:
                continue
            key, raw_value = item.split("=", 1)
            parts[key] = raw_value

        try:
            bbox = [
                int(parts["x0"]),
                int(parts["y0"]),
                int(parts["x1"]),
                int(parts["y1"]),
            ]
            class_id = int(parts["class"])
            confidence = float(parts["conf"])
        except (KeyError, TypeError, ValueError):
            LOGGER.warning("Skipping malformed CSV detection chunk: %s", chunk)
            continue

        detections.append({
            "bbox": bbox,
            "class_id": class_id,
            "confidence": confidence,
        })

    return detections


def _coerce_csv_row(row):
    if not row:
        return None

    coerced = dict(row)
    for key in ("frame", "total_unique_objects"):
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

    return coerced


@app.route("/")
def index():
    """Serve the main dashboard."""
    return send_file("index.html")


@app.route("/results")
def results_page():
    """Serve the results UI."""
    return send_file("results.html")


@app.route("/models")
def models_page():
    """Serve the models catalog UI."""
    return send_file("models.html")


@app.route("/api/config")
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
    return jsonify({"stream_port": 8889})


@app.route("/api/grid", methods=["GET"])
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
    return jsonify(get_grid_api_state())


@app.route("/api/grid", methods=["PUT"])
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
    payload = request.get_json(silent=True) or {}
    if not apply_grid_api_update(payload):
        return jsonify({"error": "Missing grid settings"}), 400

    return jsonify(get_grid_api_state())


@app.route("/api/models")
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
    """
    try:
        models = []
        # Search in UL and RF folders for engine files
        search_paths = [
            os.path.join(MODEL_DIR, "ul", "*.engine"),
            os.path.join(MODEL_DIR, "rf", "*.engine")
        ]
        
        for p in search_paths:
            files = glob.glob(p)
            for f in files:
                parent = os.path.basename(os.path.dirname(f))
                name = os.path.basename(f)
                
                models.append({
                    "path": f,
                    "name": name,
                    "type": parent,
                    "display": f"[{parent.upper()}] {name}"
                })
        
        return jsonify({"models": models})
    except Exception as e:
        LOGGER.error(f"Failed to list models: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-catalog")
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
                  compiled:
                    type: boolean
    """
    try:
        return jsonify({"models": build_model_catalog(), "tensorrt": {"current": get_current_tensorrt_version()}})
    except Exception as e:
        LOGGER.error(f"Failed to list model catalog: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/model-upload", methods=["POST"])
def api_model_upload():
    """
    Upload a model weights file into the catalog.
    ---
    tags:
      - Configuration
    consumes:
      - multipart/form-data
    responses:
      200:
        description: Model uploaded successfully
      400:
        description: Invalid upload request
    """
    model_type = (request.form.get("type") or "").strip().lower()
    if model_type not in ("ul", "rf"):
        return jsonify({"error": "Invalid model type."}), 400

    uploaded_file = request.files.get("file")
    if uploaded_file is None or not uploaded_file.filename:
        return jsonify({"error": "Missing model file."}), 400

    filename = _sanitize_uploaded_model_filename(uploaded_file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename."}), 400

    ext = os.path.splitext(filename)[1].lower()
    if ext not in MODEL_UPLOAD_EXTENSIONS:
        return jsonify({"error": "Only .pt model weights are supported."}), 400

    target_dir = os.path.join(MODEL_DIR, model_type)
    os.makedirs(target_dir, exist_ok=True)
    target_path = os.path.join(target_dir, filename)
    if os.path.exists(target_path):
        return jsonify({"error": "A model with this filename already exists."}), 409

    try:
        uploaded_file.save(target_path)
    except Exception as exc:
        LOGGER.error("Failed to save uploaded model %s: %s", target_path, exc)
        try:
            if os.path.exists(target_path):
                os.remove(target_path)
        except OSError:
            pass
        return jsonify({"error": "Failed to store uploaded model."}), 500

    return jsonify(
        {
            "uploaded": {
                "type": model_type,
                "name": os.path.splitext(filename)[0],
                "path": target_path,
            },
            "tensorrt": {"current": get_current_tensorrt_version()},
        }
    )


@app.route("/api/model-compile", methods=["POST"])
def api_model_compile():
    """
    Start async model compilation to TensorRT (FP16).
    ---
    tags:
      - Configuration
    consumes:
      - application/json
    responses:
      200:
        description: Compile job queued
    """
    payload = request.get_json(silent=True) or {}
    model_type = payload.get("type")
    model_name = payload.get("name")

    if model_type not in ("ul", "rf") or not model_name:
        return jsonify({"error": "Invalid model type or name."}), 400

    catalog = build_model_catalog()
    entry = next(
        (item for item in catalog if item["type"] == model_type and item["name"] == model_name),
        None,
    )
    if entry is None:
        return jsonify({"error": "Model not found."}), 404

    if entry.get("compiled"):
        return jsonify({"error": "Model already compiled."}), 400

    with compile_jobs_lock:
        existing_job = _find_active_job_for_model(model_type, model_name)
        if existing_job is not None:
            return jsonify({"job_id": existing_job["id"], "already_running": True})

    source_paths = entry.get("source_paths", [])
    source_pt = next((p for p in source_paths if p.endswith(".pt")), None)
    if not source_pt:
        return jsonify({"error": "No .pt source found for compilation."}), 400

    if model_type == "ul":
        command = [
            "yolo",
            "export",
            "format=engine",
            f"model={source_pt}",
            "imgsz=640",
            "half",
        ]
    else:
        command = [sys.executable, "/app/convert.py", "--model", source_pt]

    job_id = uuid.uuid4().hex
    with compile_jobs_lock:
        compile_jobs[job_id] = {
            "id": job_id,
            "status": "queued",
            "logs": deque(maxlen=500),
            "created_at": datetime.utcnow().isoformat() + "Z",
            "model": {
                "type": model_type,
                "name": model_name,
                "source": source_pt,
            },
        }

    thread = threading.Thread(
        target=_run_compile_job,
        args=(job_id, command, "/app"),
        daemon=True,
    )
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/model-task", methods=["POST"])
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
    payload = request.get_json(silent=True) or {}
    model_type = payload.get("type")
    model_name = payload.get("name")
    task = payload.get("task")

    if model_type != "ul" or not model_name:
        return jsonify({"error": "Invalid model type or name."}), 400

    catalog = build_model_catalog()
    entry = next(
        (item for item in catalog if item["type"] == model_type and item["name"] == model_name),
        None,
    )
    if entry is None:
        return jsonify({"error": "Model not found."}), 404

    try:
        selected_task = set_catalog_model_task(model_type, model_name, task)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "type": model_type,
        "name": model_name,
        "task": selected_task,
    })


@app.route("/api/model-delete", methods=["POST"])
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
    payload = request.get_json(silent=True) or {}
    model_type = payload.get("type")
    model_name = payload.get("name")

    if model_type not in ("ul", "rf") or not model_name:
        return jsonify({"error": "Invalid model type or name."}), 400

    catalog = build_model_catalog()
    entry = next(
        (item for item in catalog if item["type"] == model_type and item["name"] == model_name),
        None,
    )
    if entry is None:
        return jsonify({"error": "Model not found."}), 404

    artifact_paths = collect_model_artifact_paths(entry)
    current_model_path = current_config.get("model_path")
    if is_pipeline_running() and current_model_path in artifact_paths:
        return jsonify({"error": "Cannot delete the model that is currently running."}), 409

    deleted = []
    for path in artifact_paths:
        try:
            os.remove(path)
            deleted.append(path)
        except FileNotFoundError:
            continue
        except Exception as exc:
            LOGGER.error("Failed to delete model artifact %s: %s", path, exc)
            return jsonify({"error": f"Failed to delete {path}: {exc}"}), 500

    clear_catalog_model_task(model_type, model_name)

    return jsonify({
        "type": model_type,
        "name": model_name,
        "deleted": deleted,
    })


@app.route("/api/model-compile-jobs")
def api_model_compile_jobs():
    """
    List compile jobs for UI restore after close/refresh.
    ---
    tags:
      - Configuration
    responses:
      200:
        description: Compile job list
    """
    with compile_jobs_lock:
        jobs = []
        for job in compile_jobs.values():
            jobs.append(
                {
                    "id": job["id"],
                    "status": job.get("status"),
                    "created_at": job.get("created_at"),
                    "started_at": job.get("started_at"),
                    "finished_at": job.get("finished_at"),
                    "returncode": job.get("returncode"),
                    "model": job.get("model"),
                }
            )
    jobs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return jsonify({"jobs": jobs})


@app.route("/api/model-compile/<job_id>")
def api_model_compile_status(job_id):
    """
    Get compile job status and logs.
    ---
    tags:
      - Configuration
    responses:
      200:
        description: Compile job status
    """
    with compile_jobs_lock:
        job = compile_jobs.get(job_id)
        if job is None:
            return jsonify({"error": "Job not found."}), 404

        return jsonify(
            {
                "id": job["id"],
                "status": job["status"],
                "created_at": job.get("created_at"),
                "started_at": job.get("started_at"),
                "finished_at": job.get("finished_at"),
                "returncode": job.get("returncode"),
                "command": job.get("command"),
                "model": job.get("model"),
                "logs": list(job.get("logs", [])),
            }
        )


@app.route("/api/cameras")
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
    """
    try:
        cams = CameraManager.get_available_cameras()
        return jsonify({"cameras": cams})
    except Exception as e:
        LOGGER.error(f"Failed to list cameras: {e}")
        return jsonify({"error": str(e)}), 500


def check_port(host, port, timeout=0.1):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (OSError, ConnectionRefusedError):
        return False


@app.route("/api/status")
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

    return jsonify({
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
                  video_size:
                    type: string
    """
    try:
        selected_date = _parse_results_date_filter()
        results_list = _collect_results_metadata(selected_date=selected_date)
        for item in results_list:
            item.pop("_mtime", None)
            item.pop("run_path", None)
            item.pop("csv_path", None)
            item.pop("video_path", None)

        return jsonify({"results": results_list})

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as e:
        LOGGER.error(f"Failed to list results: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/results/search")
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
        analysis_id = request.args.get('analysis_id', '').strip()
        selected_date = _parse_results_date_filter()
        results_list = []
        host_url = request.host_url.rstrip('/')

        for item in _collect_results_metadata(
            analysis_prefix=analysis_id,
            selected_date=selected_date,
        ):
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

        return jsonify({"results": results_list})

    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as e:
        LOGGER.error(f"Failed to search results: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/results/<pid>/last-row")
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
                total_unique_objects:
                  type: integer
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
                      confidence:
                        type: number
      400:
        description: Invalid analysis ID
      404:
        description: CSV file or row not found
    """
    try:
        if not pid or ".." in pid or "/" in pid:
            return jsonify({"error": "Invalid ID"}), 400

        csv_path = os.path.join(HQ_OUTPUT_DIR, pid, f"{pid}.csv")
        if not os.path.isfile(csv_path):
            return jsonify({"error": "CSV not found"}), 404

        row = _read_last_csv_row(csv_path)
        if row is None:
            return jsonify({"error": "CSV is empty"}), 404

        return jsonify({
            "analysis_id": pid,
            "row": _coerce_csv_row(row),
        })

    except Exception as exc:
        LOGGER.error("Failed to read last CSV row for %s: %s", pid, exc)
        return jsonify({"error": str(exc)}), 500


@app.route("/api/results/<pid>", methods=["DELETE"])
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
    """
    try:
        if not pid or ".." in pid or "/" in pid:
            return jsonify({"error": "Invalid ID"}), 400

        run_dir = os.path.join(HQ_OUTPUT_DIR, pid)
        deleted_files = []

        if os.path.isdir(run_dir):
            for root, _, files in os.walk(run_dir):
                for name in files:
                    rel_path = os.path.relpath(os.path.join(root, name), HQ_OUTPUT_DIR)
                    deleted_files.append(rel_path)
            shutil.rmtree(run_dir)

        LOGGER.info(f"Deleted results for ID {pid}: {deleted_files}")
        return jsonify({"success": True, "deleted": deleted_files})

    except Exception as e:
        LOGGER.error(f"Failed to delete results for {pid}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/download/<path:filename>")
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
      404:
        description: Not found
    """
    try:
        return send_from_directory(HQ_OUTPUT_DIR, filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/vendor/<path:filename>")
def vendor_file(filename):
    """Serve bundled frontend assets."""
    try:
        return send_from_directory(VENDOR_DIR, filename)
    except Exception as e:
        return jsonify({"error": str(e)}), 404


@app.route("/api/results/<pid>/download")
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
      404:
        description: Not found
    """
    try:
        if not pid or ".." in pid or "/" in pid:
            return jsonify({"error": "Invalid ID"}), 400

        run_dir = os.path.join(HQ_OUTPUT_DIR, pid)
        if not os.path.isdir(run_dir):
            return jsonify({"error": "Not found"}), 404

        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(run_dir):
                for name in files:
                    full_path = os.path.join(root, name)
                    rel_path = os.path.relpath(full_path, run_dir)
                    zf.write(full_path, rel_path)

        archive.seek(0)
        return send_file(
            archive,
            mimetype="application/zip",
            as_attachment=True,
            download_name=f"{pid}.zip",
        )
    except Exception as e:
        LOGGER.error(f"Failed to create zip for {pid}: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/start", methods=["POST"])
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
            grid_count_enabled:
              type: boolean
              description: Optional initial grid feature state. When enabled, grid detection runs and counting is limited to the detected viewport.
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
    """
    global pipeline, last_error, current_config, pipeline_id

    if is_pipeline_running():
        return jsonify({"error": "already running"}), 400

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
        data = request.json or {}
        
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
            pipeline_id = analysis_num
        else:
            start_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            pipeline_id = f"{start_time_str}-{uuid.uuid4().hex}"
        
        model_path = data.get("model_path", None)
        
        if not model_path:
            model_path = "/app/model/weights-fp16.engine"

        requested_model_task = _sanitize_ul_model_task(data.get("model_task"))
        resolved_model_task = requested_model_task or resolve_model_task_for_path(model_path)

        requested_conf = float(data.get("vis_conf", 0.75))
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
        ))

        if source_type == "camera":
            device = data.get("device")
            width = int(data.get("width", 1280))
            height = int(data.get("height", 720))
            fps = int(data.get("fps", 30))
            pixel_format = data.get("format", "MJPG")
            
            if not device:
                raise ValueError("No device selected")

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
                
            if not os.path.isfile(video):
                raise ValueError("File not found: %s" % video)
                
            args_dict["mode"] = "file"
            video_desc = os.path.basename(video)
            args_dict["fps"] = 30 
            
            args = argparse.Namespace(**args_dict)
            reader = FileReader(video, args.fps)
            
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
        }

        return jsonify({"success": True, "pipeline_id": pipeline_id})

    except Exception as e:
        last_error = str(e)
        LOGGER.exception("Failed to start pipeline: %s", e)
        if tmp_pipeline is not None:
            try:
                tmp_pipeline.__exit__(type(e), e, e.__traceback__)
            except Exception:
                pass
        return jsonify({"error": str(e)}), 500


@app.route("/api/stop", methods=["POST"])
def api_stop():
    """
    Stop the inference pipeline.
    ---
    tags:
      - Control
    responses:
      200:
        description: Pipeline stopped
    """
    global pipeline, last_error, current_config, pipeline_id

    if pipeline is None and not is_pipeline_running():
        current_config = {}
        return jsonify({"success": True})

    try:
        if pipeline is not None:
            pipeline.__exit__(None, None, None)

        pipeline, pipeline_id = None, None
        current_config = {}

        return jsonify({"success": True})
    except Exception as e:
        last_error = str(e)
        LOGGER.exception("Failed to stop pipeline: %s", e)
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
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
    """
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "no filename"}), 400

    os.makedirs("uploads", exist_ok=True)
    path = os.path.join("uploads", file.filename)
    file.save(path)

    return jsonify({"video": path})


@app.route("/<path:path>/whep", methods=["GET", "POST", "OPTIONS"])
def proxy_whep(path):
    # Proxy WHEP requests to the local mediamtx server
    target_url = "http://127.0.0.1:8889/%s/whep" % path
    headers = {k: v for k, v in request.headers.items() if k.lower() != "host"}

    if request.method == "OPTIONS":
        resp = requests.options(target_url, headers=headers)
    elif request.method == "POST":
        resp = requests.post(target_url, data=request.data, headers=headers)
    else:
        resp = requests.get(target_url, headers=headers)

    excluded_headers = {"content-encoding", "content-length", "transfer-encoding", "connection"}
    response_headers = [(name, value) for (name, value) in resp.raw.headers.items() if name.lower() not in excluded_headers]
    return Response(resp.content, resp.status_code, response_headers)


if __name__ == "__main__":
    logging.basicConfig(level="INFO", format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("ultralytics").setLevel(logging.ERROR)
    app.run(host="0.0.0.0", port=8000, threaded=True)
