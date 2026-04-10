import glob
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import uuid

from collections import deque
from datetime import datetime


class ModelManagerError(Exception):
    pass


class ModelValidationError(ModelManagerError):
    pass


class ModelNotFoundError(ModelManagerError):
    pass


class ModelConflictError(ModelManagerError):
    pass


class ModelManager:
    SUPPORTED_MODEL_TYPES = ("ul", "rf")
    SOURCE_MODEL_EXTENSIONS = (".onnx", ".pt")
    VALID_UL_MODEL_TASKS = ("segment", "detect", "auto")
    MODEL_UPLOAD_EXTENSIONS = (".pt",)
    MODEL_COMPILE_METADATA_SUFFIX = ".compile.json"
    DEFAULT_MODEL_CONFIDENCE_THRESHOLD = 0.75
    COMPILE_JOB_LOG_LIMIT = 500

    def __init__(
        self,
        *,
        auth_module,
        logger=None,
        model_dir="/app/model",
        compile_cwd="/app",
        convert_script="/app/convert.py",
    ):
        self.auth = auth_module
        self.logger = logger or logging.getLogger("app")
        self.model_dir = model_dir
        self.model_metadata_filepath = os.path.join(self.model_dir, "model_metadata.json")
        self.model_tasks_filepath = os.path.join(self.model_dir, "model_tasks.json")
        self.compile_cwd = compile_cwd
        self.convert_script = convert_script
        self.compile_jobs = {}
        self.compile_jobs_lock = threading.Lock()
        self.model_metadata_lock = threading.Lock()
        self.model_task_overrides_lock = threading.Lock()

    def _utcnow_text(self):
        return datetime.utcnow().isoformat() + "Z"

    def _model_dir_for_type(self, model_type):
        return os.path.join(self.model_dir, model_type)

    def _catalog_key(self, model_type, model_name):
        return model_type, model_name

    def _model_name_candidates(self, model_name):
        candidates = [model_name]
        if model_name.endswith("-fp16"):
            candidates.append(model_name[:-5])
        return tuple(dict.fromkeys(candidate for candidate in candidates if candidate))

    def _new_catalog_entry(self, model_type, model_name):
        return dict(
            name=model_name,
            type=model_type,
            sources=[],
            source_paths=[],
        )

    def _get_or_create_catalog_entry(self, model_map, model_type, model_name):
        key = self._catalog_key(model_type, model_name)
        entry = model_map.get(key)
        if entry is None:
            entry = self._new_catalog_entry(model_type, model_name)
            model_map[key] = entry
        return entry

    def _new_engine_info(self, engine_path):
        return dict(
            name=os.path.basename(engine_path),
            path=engine_path,
        )

    def _new_job_model(self, model_type, model_name, source_path):
        return dict(
            type=model_type,
            name=model_name,
            source=source_path,
        )

    def _new_compile_job(self, job_id, model_type, model_name, source_path):
        return dict(
            id=job_id,
            status="queued",
            logs=deque(maxlen=self.COMPILE_JOB_LOG_LIMIT),
            created_at=self._utcnow_text(),
            model=self._new_job_model(model_type, model_name, source_path),
        )

    def _serialize_compile_job(self, job, *, include_logs=False):
        payload = dict(
            id=job["id"],
            status=job.get("status"),
            created_at=job.get("created_at"),
            started_at=job.get("started_at"),
            finished_at=job.get("finished_at"),
            returncode=job.get("returncode"),
            model=job.get("model"),
        )
        if include_logs:
            payload["command"] = job.get("command")
            payload["logs"] = list(job.get("logs", []))
        return payload

    def _validate_model_type(
        self,
        model_type,
        *,
        allowed_types=None,
        error_message="Invalid model type.",
    ):
        allowed_types = tuple(allowed_types or self.SUPPORTED_MODEL_TYPES)
        if model_type not in allowed_types:
            raise ModelValidationError(error_message)

    def _validate_model_reference(self, model_type, model_name, *, allowed_types=None):
        self._validate_model_type(
            model_type,
            allowed_types=allowed_types,
            error_message="Invalid model type or name.",
        )
        if not model_name:
            raise ModelValidationError("Invalid model type or name.")

    def _get_catalog_entry(self, model_type, model_name):
        catalog = self.build_model_catalog()
        return next(
            (
                item
                for item in catalog
                if item["type"] == model_type and item["name"] == model_name
            ),
            None,
        )

    def _require_catalog_entry(self, model_type, model_name, *, allowed_types=None):
        self._validate_model_reference(
            model_type,
            model_name,
            allowed_types=allowed_types,
        )
        entry = self._get_catalog_entry(model_type, model_name)
        if entry is None:
            raise ModelNotFoundError("Model not found.")
        return entry

    def _append_compile_log(self, job_id, line):
        with self.compile_jobs_lock:
            job = self.compile_jobs.get(job_id)
            if job is None:
                return
            job["logs"].append(line)

    def _find_active_job_for_model(self, model_type, model_name):
        active_states = {"queued", "running"}
        selected = None
        for job in self.compile_jobs.values():
            model = job.get("model") or {}
            if model.get("type") != model_type or model.get("name") != model_name:
                continue
            if job.get("status") not in active_states:
                continue
            created = job.get("created_at", "")
            if selected is None or created > selected.get("created_at", ""):
                selected = job
        return selected

    def _model_task_key(self, model_type, model_name):
        return f"{model_type}:{model_name}"

    def sanitize_ul_model_task(self, task):
        if not isinstance(task, str):
            return None
        normalized = task.strip().lower()
        if normalized in self.VALID_UL_MODEL_TASKS:
            return normalized
        return None

    def _sanitize_default_confidence_threshold(self, value):
        try:
            threshold = float(value)
        except (TypeError, ValueError):
            return None
        if threshold < 0.0 or threshold > 1.0:
            return None
        return threshold

    def _sanitize_model_metadata_entry(self, value):
        if not isinstance(value, dict):
            return {}

        entry = {}
        default_threshold = self._sanitize_default_confidence_threshold(
            value.get("default_confidence_threshold")
        )
        if default_threshold is not None:
            entry["default_confidence_threshold"] = default_threshold
        return entry

    def _read_model_metadata_unlocked(self):
        if not os.path.exists(self.model_metadata_filepath):
            return {}

        try:
            with open(self.model_metadata_filepath, "r", encoding="utf-8") as metadata_input:
                data = json.load(metadata_input)
        except FileNotFoundError:
            return {}
        except Exception as exc:
            self.logger.error(
                "Failed to read model metadata %s: %s",
                self.model_metadata_filepath,
                exc,
            )
            return {}

        if not isinstance(data, dict):
            return {}

        metadata = {}
        for key, value in data.items():
            if not isinstance(key, str):
                continue
            entry = self._sanitize_model_metadata_entry(value)
            if entry:
                metadata[key] = entry
        return metadata

    def _write_model_metadata_unlocked(self, metadata):
        sanitized = {}
        for key, value in (metadata or {}).items():
            if not isinstance(key, str):
                continue
            entry = self._sanitize_model_metadata_entry(value)
            if entry:
                sanitized[key] = entry

        os.makedirs(os.path.dirname(self.model_metadata_filepath), exist_ok=True)
        tmp_path = f"{self.model_metadata_filepath}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as metadata_output:
            json.dump(sanitized, metadata_output, indent=2, sort_keys=True)
            metadata_output.write("\n")
        os.replace(tmp_path, self.model_metadata_filepath)

    def _read_model_task_overrides_unlocked(self):
        if not os.path.exists(self.model_tasks_filepath):
            return {}

        try:
            with open(self.model_tasks_filepath, "r", encoding="utf-8") as task_input:
                data = json.load(task_input)
        except FileNotFoundError:
            return {}
        except Exception as exc:
            self.logger.error(
                "Failed to read model task config %s: %s",
                self.model_tasks_filepath,
                exc,
            )
            return {}

        if not isinstance(data, dict):
            return {}

        overrides = {}
        for key, value in data.items():
            if not isinstance(key, str):
                continue
            task = self.sanitize_ul_model_task(value)
            if task is not None:
                overrides[key] = task
        return overrides

    def _write_model_task_overrides_unlocked(self, overrides):
        sanitized = {}
        for key, value in (overrides or {}).items():
            if not isinstance(key, str):
                continue
            task = self.sanitize_ul_model_task(value)
            if task is not None:
                sanitized[key] = task

        os.makedirs(os.path.dirname(self.model_tasks_filepath), exist_ok=True)
        tmp_path = f"{self.model_tasks_filepath}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as task_output:
            json.dump(sanitized, task_output, indent=2, sort_keys=True)
            task_output.write("\n")
        os.replace(tmp_path, self.model_tasks_filepath)

    def get_catalog_model_task(self, model_type, model_name):
        if model_type != "ul":
            return None

        with self.model_task_overrides_lock:
            overrides = self._read_model_task_overrides_unlocked()
        return overrides.get(self._model_task_key(model_type, model_name), "segment")

    def _set_catalog_model_task(self, model_type, model_name, task):
        if model_type != "ul":
            raise ModelValidationError(
                "Task override is only supported for Ultralytics models."
            )

        normalized_task = self.sanitize_ul_model_task(task)
        if normalized_task is None:
            raise ModelValidationError("Invalid model task.")

        with self.model_task_overrides_lock:
            overrides = self._read_model_task_overrides_unlocked()
            key = self._model_task_key(model_type, model_name)
            if normalized_task == "segment":
                overrides.pop(key, None)
            else:
                overrides[key] = normalized_task
            self._write_model_task_overrides_unlocked(overrides)

        return normalized_task

    def clear_catalog_model_task(self, model_type, model_name):
        with self.model_task_overrides_lock:
            overrides = self._read_model_task_overrides_unlocked()
            changed = False
            for candidate in self._model_name_candidates(model_name):
                key = self._model_task_key(model_type, candidate)
                if key in overrides:
                    overrides.pop(key, None)
                    changed = True
            if changed:
                self._write_model_task_overrides_unlocked(overrides)

    def get_catalog_model_default_confidence_threshold(self, model_type, model_name):
        with self.model_metadata_lock:
            metadata = self._read_model_metadata_unlocked()

        for candidate in self._model_name_candidates(model_name):
            entry = metadata.get(self._model_task_key(model_type, candidate)) or {}
            default_threshold = entry.get("default_confidence_threshold")
            if default_threshold is not None:
                return default_threshold
        return self.DEFAULT_MODEL_CONFIDENCE_THRESHOLD

    def set_model_default_confidence_threshold(self, model_type, model_name, value):
        self._require_catalog_entry(model_type, model_name)

        normalized_threshold = self._sanitize_default_confidence_threshold(value)
        if normalized_threshold is None:
            raise ModelValidationError("Invalid default confidence threshold.")

        with self.model_metadata_lock:
            metadata = self._read_model_metadata_unlocked()
            key = self._model_task_key(model_type, model_name)
            entry = dict(metadata.get(key) or {})

            if normalized_threshold == self.DEFAULT_MODEL_CONFIDENCE_THRESHOLD:
                entry.pop("default_confidence_threshold", None)
            else:
                entry["default_confidence_threshold"] = normalized_threshold

            if entry:
                metadata[key] = entry
            else:
                metadata.pop(key, None)
            self._write_model_metadata_unlocked(metadata)

        return dict(
            type=model_type,
            name=model_name,
            default_confidence_threshold=normalized_threshold,
        )

    def clear_catalog_model_metadata(self, model_type, model_name):
        with self.model_metadata_lock:
            metadata = self._read_model_metadata_unlocked()
            changed = False
            for candidate in self._model_name_candidates(model_name):
                key = self._model_task_key(model_type, candidate)
                if key in metadata:
                    metadata.pop(key, None)
                    changed = True
            if changed:
                self._write_model_metadata_unlocked(metadata)

    def resolve_model_task_for_path(self, model_path):
        if not model_path:
            return "segment"

        normalized = os.path.normpath(model_path)
        try:
            relative = os.path.relpath(normalized, self.model_dir)
        except ValueError:
            return "segment"

        parts = relative.split(os.sep)
        if len(parts) < 2:
            return "segment"

        model_type = parts[0]
        if model_type != "ul":
            return "auto"

        base_name = os.path.splitext(parts[-1])[0]
        with self.model_task_overrides_lock:
            overrides = self._read_model_task_overrides_unlocked()

        for candidate in self._model_name_candidates(base_name):
            task = overrides.get(self._model_task_key(model_type, candidate))
            if task is not None:
                return task
        return "segment"

    def resolve_model_default_confidence_threshold_for_path(self, model_path, fallback=None):
        fallback_threshold = self.DEFAULT_MODEL_CONFIDENCE_THRESHOLD
        if fallback is not None:
            sanitized_fallback = self._sanitize_default_confidence_threshold(fallback)
            if sanitized_fallback is not None:
                fallback_threshold = sanitized_fallback

        if not model_path:
            return fallback_threshold

        normalized = os.path.normpath(model_path)
        try:
            relative = os.path.relpath(normalized, self.model_dir)
        except ValueError:
            return fallback_threshold

        parts = relative.split(os.sep)
        if len(parts) < 2:
            return fallback_threshold

        model_type = parts[0]
        base_name = os.path.splitext(parts[-1])[0]
        return self.get_catalog_model_default_confidence_threshold(model_type, base_name)

    def get_current_tensorrt_version(self):
        try:
            import tensorrt as trt

            return str(trt.__version__)
        except Exception:
            return None

    def _compile_metadata_path(self, engine_path):
        if not engine_path:
            return None
        return f"{engine_path}{self.MODEL_COMPILE_METADATA_SUFFIX}"

    def _read_compile_metadata(self, engine_path):
        metadata_path = self._compile_metadata_path(engine_path)
        if not metadata_path or not os.path.isfile(metadata_path):
            return {}
        try:
            with open(metadata_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            return data if isinstance(data, dict) else {}
        except Exception as exc:
            self.logger.warning(
                "Failed to read compile metadata %s: %s",
                metadata_path,
                exc,
            )
            return {}

    def _write_compile_metadata(
        self,
        engine_path,
        *,
        model_type=None,
        source_path=None,
        command=None,
    ):
        metadata_path = self._compile_metadata_path(engine_path)
        if not metadata_path:
            return

        payload = dict(
            tensorrt_version=self.get_current_tensorrt_version(),
            compiled_at=self._utcnow_text(),
        )
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
            self.logger.warning(
                "Failed to write compile metadata %s: %s",
                metadata_path,
                exc,
            )

    def _resolve_compiled_engine_path(self, model_type, source_path):
        if not source_path:
            return None

        source_dir = os.path.dirname(source_path)
        source_base = os.path.splitext(os.path.basename(source_path))[0]
        if model_type == "ul":
            candidate_names = (f"{source_base}.engine",)
        elif model_type == "rf":
            candidate_names = (
                f"{source_base}-fp16.engine",
                f"{source_base}.engine",
            )
        else:
            candidate_names = (
                f"{source_base}.engine",
                f"{source_base}-fp16.engine",
            )

        for candidate_name in candidate_names:
            candidate_path = os.path.join(source_dir, candidate_name)
            if os.path.isfile(candidate_path):
                return candidate_path
        return None

    def _record_compile_metadata(self, job_id):
        with self.compile_jobs_lock:
            job = self.compile_jobs.get(job_id)
            if job is None:
                return
            model = job.get("model") or {}
            source_path = model.get("source")
            model_type = model.get("type")
            command = job.get("command")

        engine_path = self._resolve_compiled_engine_path(model_type, source_path)
        if not engine_path:
            return

        self._write_compile_metadata(
            engine_path,
            model_type=model_type,
            source_path=source_path,
            command=command,
        )
        self._append_compile_log(
            job_id,
            f"Saved compile metadata: {self._compile_metadata_path(engine_path)}",
        )

    def _cleanup_ul_compile_intermediates(self, job_id):
        with self.compile_jobs_lock:
            job = self.compile_jobs.get(job_id)
            if job is None:
                return
            model = job.get("model") or {}
            source_path = model.get("source")
            model_type = model.get("type")
            started_ts = job.get("started_ts", time.time())

        if model_type != "ul" or not source_path:
            return

        source_dir = os.path.dirname(source_path)
        source_base = os.path.splitext(os.path.basename(source_path))[0]
        patterns = (
            os.path.join(source_dir, f"{source_base}*.onnx"),
            os.path.join("/app/runs", "**", f"{source_base}*.onnx"),
        )

        candidates = set()
        for pattern in patterns:
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
                self._append_compile_log(job_id, f"Removed intermediate artifact: {path}")
            except Exception as exc:
                self._append_compile_log(
                    job_id,
                    f"Failed to remove intermediate artifact {path}: {exc}",
                )

    def _sanitize_uploaded_model_filename(self, filename):
        safe_name = os.path.basename((filename or "").strip())
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", safe_name)
        return safe_name.strip("._")

    def _cleanup_partial_upload(self, target_path):
        if not target_path or not os.path.exists(target_path):
            return
        try:
            os.remove(target_path)
        except OSError as exc:
            self.logger.warning(
                "Failed to remove partial upload %s: %s",
                target_path,
                exc,
            )

    def _find_catalog_engine(self, model_type, model_name):
        base_dir = self._model_dir_for_type(model_type)
        candidate_paths = [os.path.join(base_dir, f"{model_name}.engine")]
        if not model_name.endswith("-fp16"):
            candidate_paths.append(os.path.join(base_dir, f"{model_name}-fp16.engine"))

        for candidate_path in candidate_paths:
            if not os.path.exists(candidate_path):
                continue
            engine = self._new_engine_info(candidate_path)
            metadata = self._read_compile_metadata(candidate_path)
            if metadata:
                engine.update(
                    dict(
                        tensorrt_version=metadata.get("tensorrt_version"),
                        compiled_at=metadata.get("compiled_at"),
                    )
                )
            return engine
        return None

    def _build_compile_command(self, model_type, source_path):
        if model_type == "ul":
            return [
                "yolo",
                "export",
                "format=engine",
                f"model={source_path}",
                "imgsz=640",
                "half",
            ]
        return [sys.executable, self.convert_script, "--model", source_path]

    def _find_compile_source_path(self, entry):
        return next(
            (
                path
                for path in (entry.get("source_paths") or [])
                if path.endswith(".pt")
            ),
            None,
        )

    def collect_model_artifact_paths(self, entry):
        if not entry:
            return []

        base_dir = self._model_dir_for_type(entry["type"])
        base_name = entry["name"]
        artifacts = set(entry.get("source_paths") or [])

        engine_path = (entry.get("engine") or {}).get("path")
        if engine_path:
            artifacts.add(engine_path)

        artifacts.update(
            os.path.join(base_dir, f"{base_name}{suffix}")
            for suffix in (".pt", ".onnx", ".engine")
        )
        if not base_name.endswith("-fp16"):
            artifacts.add(os.path.join(base_dir, f"{base_name}-fp16.engine"))

        artifacts.update(
            metadata_path
            for metadata_path in (
                self._compile_metadata_path(path)
                for path in list(artifacts)
                if path.endswith(".engine")
            )
            if metadata_path
        )

        return sorted(path for path in artifacts if os.path.isfile(path))

    def _ensure_ul_engine_output(self, job_id):
        with self.compile_jobs_lock:
            job = self.compile_jobs.get(job_id)
            if job is None:
                return False
            model = job.get("model") or {}
            source_path = model.get("source")

        if not source_path:
            return False

        expected_engine = self._resolve_compiled_engine_path("ul", source_path)
        if expected_engine and os.path.exists(expected_engine):
            self._append_compile_log(job_id, f"Engine ready: {expected_engine}")
            return True

        source_dir = os.path.dirname(source_path)
        source_base = os.path.splitext(os.path.basename(source_path))[0]
        expected_engine = os.path.join(source_dir, f"{source_base}.engine")
        self._append_compile_log(
            job_id,
            f"Compile finished, but expected engine artifact was not found: {expected_engine}",
        )
        return False

    def _run_compile_job(self, job_id, command, cwd=None):
        with self.compile_jobs_lock:
            job = self.compile_jobs.get(job_id)
            if job is None:
                return
            job["status"] = "running"
            job["started_at"] = self._utcnow_text()
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
            self._append_compile_log(job_id, f"Failed to start compile: {exc}")
            with self.compile_jobs_lock:
                job = self.compile_jobs.get(job_id)
                if job is not None:
                    job["status"] = "error"
                    job["finished_at"] = self._utcnow_text()
                    job["returncode"] = None
            return

        total_lines = 0
        last_heartbeat = time.time()
        try:
            for line in process.stdout:
                if not line:
                    continue

                total_lines += 1
                self._append_compile_log(job_id, line.rstrip())

                now = time.time()
                if now - last_heartbeat < 15:
                    continue

                with self.compile_jobs_lock:
                    job = self.compile_jobs.get(job_id)
                    started_ts = (job or {}).get("started_ts", now)
                elapsed = int(now - started_ts)
                self._append_compile_log(
                    job_id,
                    f"[still running] {elapsed}s elapsed, processed {total_lines} output lines",
                )
                last_heartbeat = now
        except Exception as exc:
            self._append_compile_log(job_id, f"Error reading output: {exc}")

        process.wait()
        with self.compile_jobs_lock:
            model_type = (self.compile_jobs.get(job_id) or {}).get("model", {}).get("type")

        compile_artifacts_ready = process.returncode == 0
        if process.returncode == 0:
            if model_type == "ul":
                compile_artifacts_ready = self._ensure_ul_engine_output(job_id)
                if compile_artifacts_ready:
                    self._cleanup_ul_compile_intermediates(job_id)
            if compile_artifacts_ready:
                self._record_compile_metadata(job_id)

        with self.compile_jobs_lock:
            job = self.compile_jobs.get(job_id)
            if job is not None:
                job["returncode"] = process.returncode
                job["finished_at"] = self._utcnow_text()
                job["status"] = "done" if compile_artifacts_ready else "error"

    def list_engine_models(self):
        models = []
        for model_type in self.SUPPORTED_MODEL_TYPES:
            pattern = os.path.join(self._model_dir_for_type(model_type), "*.engine")
            for path in sorted(glob.glob(pattern)):
                name = os.path.basename(path)
                models.append(
                    dict(
                        path=path,
                        name=name,
                        type=model_type,
                        display=f"[{model_type.upper()}] {name}",
                        default_confidence_threshold=self.get_catalog_model_default_confidence_threshold(
                            model_type,
                            os.path.splitext(name)[0],
                        ),
                        owner_username=self.auth.get_model_owner_username(
                            model_type,
                            os.path.splitext(name)[0],
                        ),
                    )
                )
        return models

    def build_model_catalog(self):
        model_map = {}

        for model_type in self.SUPPORTED_MODEL_TYPES:
            base_dir = self._model_dir_for_type(model_type)
            if not os.path.isdir(base_dir):
                continue

            for ext in self.SOURCE_MODEL_EXTENSIONS:
                pattern = os.path.join(base_dir, f"*{ext}")
                for path in sorted(glob.glob(pattern)):
                    base_name = os.path.splitext(os.path.basename(path))[0]
                    entry = self._get_or_create_catalog_entry(
                        model_map,
                        model_type,
                        base_name,
                    )
                    entry["sources"].append(ext.lstrip("."))
                    entry["source_paths"].append(path)

            for path in sorted(glob.glob(os.path.join(base_dir, "*.engine"))):
                base_name = os.path.splitext(os.path.basename(path))[0]
                self._get_or_create_catalog_entry(model_map, model_type, base_name)

        models = []
        for (model_type, base_name), entry in sorted(model_map.items(), key=lambda item: item[0]):
            engine = self._find_catalog_engine(model_type, base_name)
            entry["engine"] = engine
            entry["compiled"] = engine is not None
            entry["display"] = f"[{model_type.upper()}] {base_name}"
            entry["task"] = self.get_catalog_model_task(model_type, base_name)
            entry["default_confidence_threshold"] = self.get_catalog_model_default_confidence_threshold(
                model_type,
                base_name,
            )
            entry["owner_username"] = self.auth.get_model_owner_username(model_type, base_name)
            models.append(entry)

        return models

    def upload_model(self, model_type, uploaded_file, owner_user_id):
        normalized_type = str(model_type or "").strip().lower()
        self._validate_model_type(normalized_type)

        if uploaded_file is None or not getattr(uploaded_file, "filename", ""):
            raise ModelValidationError("Missing model file.")

        filename = self._sanitize_uploaded_model_filename(uploaded_file.filename)
        if not filename:
            raise ModelValidationError("Invalid filename.")

        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.MODEL_UPLOAD_EXTENSIONS:
            raise ModelValidationError("Only .pt model weights are supported.")

        target_dir = self._model_dir_for_type(normalized_type)
        os.makedirs(target_dir, exist_ok=True)
        target_path = os.path.join(target_dir, filename)
        if os.path.exists(target_path):
            raise ModelConflictError("A model with this filename already exists.")

        try:
            uploaded_file.save(target_path)
        except Exception as exc:
            self.logger.error("Failed to save uploaded model %s: %s", target_path, exc)
            self._cleanup_partial_upload(target_path)
            raise ModelManagerError("Failed to store uploaded model.") from exc

        model_name = os.path.splitext(filename)[0]
        self.auth.store_model_owner(normalized_type, model_name, owner_user_id)
        return dict(type=normalized_type, name=model_name, path=target_path)

    def start_compile(self, model_type, model_name):
        entry = self._require_catalog_entry(model_type, model_name)
        if entry.get("compiled"):
            raise ModelValidationError("Model already compiled.")

        with self.compile_jobs_lock:
            existing_job = self._find_active_job_for_model(model_type, model_name)
            if existing_job is not None:
                return dict(job_id=existing_job["id"], already_running=True)

        source_path = self._find_compile_source_path(entry)
        if not source_path:
            raise ModelValidationError("No .pt source found for compilation.")

        command = self._build_compile_command(model_type, source_path)
        job_id = uuid.uuid4().hex

        with self.compile_jobs_lock:
            self.compile_jobs[job_id] = self._new_compile_job(
                job_id,
                model_type,
                model_name,
                source_path,
            )

        thread = threading.Thread(
            target=self._run_compile_job,
            args=(job_id, command, self.compile_cwd),
            daemon=True,
        )
        thread.start()

        return dict(job_id=job_id)

    def set_model_task(self, model_type, model_name, task):
        self._require_catalog_entry(model_type, model_name, allowed_types=("ul",))
        selected_task = self._set_catalog_model_task(model_type, model_name, task)
        return dict(type=model_type, name=model_name, task=selected_task)

    def delete_model(
        self,
        model_type,
        model_name,
        *,
        current_model_path=None,
        pipeline_running=False,
    ):
        entry = self._require_catalog_entry(model_type, model_name)
        artifact_paths = self.collect_model_artifact_paths(entry)
        if pipeline_running and current_model_path in artifact_paths:
            raise ModelConflictError(
                "Cannot delete the model that is currently running."
            )

        deleted = []
        for path in artifact_paths:
            try:
                os.remove(path)
                deleted.append(path)
            except FileNotFoundError:
                continue
            except Exception as exc:
                self.logger.error("Failed to delete model artifact %s: %s", path, exc)
                raise ModelManagerError(f"Failed to delete {path}: {exc}") from exc

        self.clear_catalog_model_task(model_type, model_name)
        self.clear_catalog_model_metadata(model_type, model_name)
        self.auth.delete_model_owner(model_type, model_name)
        return dict(type=model_type, name=model_name, deleted=deleted)

    def list_compile_jobs(self):
        with self.compile_jobs_lock:
            jobs = [self._serialize_compile_job(job) for job in self.compile_jobs.values()]
        jobs.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return jobs

    def get_compile_job(self, job_id):
        with self.compile_jobs_lock:
            job = self.compile_jobs.get(job_id)
            if job is None:
                raise ModelNotFoundError("Job not found.")
            return self._serialize_compile_job(job, include_logs=True)
