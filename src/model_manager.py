import glob
import json
import logging
import os
import pickletools
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
import zipfile

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
    MODEL_UPLOAD_EXTENSIONS = (".pt", ".zip")
    RF_PACKAGE_REQUIRED_SUFFIXES = (".onnx", ".inference_config.json", ".class_names.txt")
    RF_PACKAGE_OPTIONAL_SUFFIXES = (".model_config.json", ".manifest.json")
    RF_PACKAGE_ENGINE_SUFFIXES = (
        "-fp16.engine",
        "-fp16.engine.json",
        "-fp16.engine.class_names.txt",
        "-fp16.engine.compile.json",
    )
    RF_PACKAGE_SIDECAR_SUFFIXES = (
        ".inference_config.json",
        ".model_config.json",
        ".class_names.txt",
        ".manifest.json",
    )
    MODEL_COMPILE_METADATA_SUFFIX = ".compile.json"
    DEFAULT_MODEL_CONFIDENCE_THRESHOLD = 0.75
    DEFAULT_MODEL_INFERENCE_WIDTH = 640
    DEFAULT_MODEL_INFERENCE_HEIGHT = 640
    DEFAULT_TILLETIA_FILTER_MAX_WIDTH_PX = 68
    DEFAULT_TILLETIA_FILTER_MAX_HEIGHT_PX = 68
    DEFAULT_TILLETIA_FILTER_TRAINING_WIDTH = 2592
    DEFAULT_TILLETIA_FILTER_TRAINING_HEIGHT = 1944
    TILLETIA_FILTER_METADATA_FIELDS = (
        "tilletia_filter_max_width_px",
        "tilletia_filter_max_height_px",
        "tilletia_filter_training_width",
        "tilletia_filter_training_height",
    )
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

    def _sanitize_inference_dimension(self, value):
        try:
            dimension = int(value)
        except (TypeError, ValueError):
            return None
        if dimension < 32 or dimension > 8192:
            return None
        return dimension

    def _sanitize_positive_integer(self, value):
        if isinstance(value, bool):
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not numeric.is_integer():
            return None
        normalized = int(numeric)
        return normalized if normalized > 0 else None

    def _normalize_imgsz_to_dimensions(self, value):
        if value is None:
            return None

        if isinstance(value, str):
            normalized = value.strip().strip("[]()")
            if not normalized:
                return None
            if "," in normalized:
                parts = [self._sanitize_inference_dimension(part.strip()) for part in normalized.split(",")]
                if len(parts) >= 2 and parts[0] is not None and parts[1] is not None:
                    return {
                        "inference_width": parts[1],
                        "inference_height": parts[0],
                    }
            dimension = self._sanitize_inference_dimension(normalized)
            if dimension is not None:
                return {
                    "inference_width": dimension,
                    "inference_height": dimension,
                }
            return None

        if isinstance(value, (list, tuple)):
            parts = [self._sanitize_inference_dimension(part) for part in value]
            if len(parts) >= 2 and parts[0] is not None and parts[1] is not None:
                return {
                    "inference_width": parts[1],
                    "inference_height": parts[0],
                }
            if len(parts) == 1 and parts[0] is not None:
                return {
                    "inference_width": parts[0],
                    "inference_height": parts[0],
                }
            return None

        dimension = self._sanitize_inference_dimension(value)
        if dimension is not None:
            return {
                "inference_width": dimension,
                "inference_height": dimension,
            }
        return None

    def _default_model_runtime_settings(self):
        return dict(
            inference_width=self.DEFAULT_MODEL_INFERENCE_WIDTH,
            inference_height=self.DEFAULT_MODEL_INFERENCE_HEIGHT,
            tilletia_filter_max_width_px=self.DEFAULT_TILLETIA_FILTER_MAX_WIDTH_PX,
            tilletia_filter_max_height_px=self.DEFAULT_TILLETIA_FILTER_MAX_HEIGHT_PX,
            tilletia_filter_training_width=self.DEFAULT_TILLETIA_FILTER_TRAINING_WIDTH,
            tilletia_filter_training_height=self.DEFAULT_TILLETIA_FILTER_TRAINING_HEIGHT,
        )

    def _sanitize_model_metadata_entry(self, value):
        if not isinstance(value, dict):
            return {}

        entry = {}
        default_threshold = self._sanitize_default_confidence_threshold(
            value.get("default_confidence_threshold")
        )
        if default_threshold is not None:
            entry["default_confidence_threshold"] = default_threshold

        filter_values = {
            field: self._sanitize_positive_integer(value.get(field))
            for field in self.TILLETIA_FILTER_METADATA_FIELDS
        }
        if all(filter_values[field] is not None for field in self.TILLETIA_FILTER_METADATA_FIELDS):
            entry.update(filter_values)
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

    def get_catalog_model_tilletia_filter_settings(self, model_type, model_name):
        defaults = {
            "tilletia_filter_max_width_px": self.DEFAULT_TILLETIA_FILTER_MAX_WIDTH_PX,
            "tilletia_filter_max_height_px": self.DEFAULT_TILLETIA_FILTER_MAX_HEIGHT_PX,
            "tilletia_filter_training_width": self.DEFAULT_TILLETIA_FILTER_TRAINING_WIDTH,
            "tilletia_filter_training_height": self.DEFAULT_TILLETIA_FILTER_TRAINING_HEIGHT,
        }
        with self.model_metadata_lock:
            metadata = self._read_model_metadata_unlocked()

        for candidate in self._model_name_candidates(model_name):
            entry = metadata.get(self._model_task_key(model_type, candidate)) or {}
            if all(field in entry for field in self.TILLETIA_FILTER_METADATA_FIELDS):
                return {
                    field: entry[field]
                    for field in self.TILLETIA_FILTER_METADATA_FIELDS
                }
        return defaults

    def _candidate_source_paths_for_model(self, model_type, model_name):
        base_dir = self._model_dir_for_type(model_type)
        candidates = []
        for candidate_name in self._model_name_candidates(model_name):
            for ext in self.SOURCE_MODEL_EXTENSIONS:
                candidate_path = os.path.join(base_dir, f"{candidate_name}{ext}")
                if os.path.isfile(candidate_path):
                    candidates.append(candidate_path)
        return tuple(dict.fromkeys(candidates))

    def _detect_ultralytics_imgsz_with_torch(self, source_path):
        try:
            import torch
        except Exception:
            return None

        try:
            checkpoint = torch.load(source_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(source_path, map_location="cpu")
        except Exception as exc:
            self.logger.warning("Failed to inspect checkpoint metadata for %s: %s", source_path, exc)
            return None

        candidates = []
        if isinstance(checkpoint, dict):
            candidates.extend(
                [
                    checkpoint.get("train_args"),
                    checkpoint.get("args"),
                    checkpoint.get("cfg"),
                ]
            )

        for candidate in candidates:
            imgsz = None
            if isinstance(candidate, dict):
                imgsz = candidate.get("imgsz")
            else:
                imgsz = getattr(candidate, "imgsz", None)
            dimensions = self._normalize_imgsz_to_dimensions(imgsz)
            if dimensions is not None:
                return dimensions
        return None

    def _detect_ultralytics_imgsz_from_zip(self, source_path):
        if not zipfile.is_zipfile(source_path):
            return None

        try:
            with zipfile.ZipFile(source_path) as archive:
                data_name = next((name for name in archive.namelist() if name.endswith("data.pkl")), None)
                if not data_name:
                    return None
                data = archive.read(data_name)
        except Exception as exc:
            self.logger.warning("Failed to inspect checkpoint archive for %s: %s", source_path, exc)
            return None

        try:
            recent_string = None
            for opcode, argument, _ in pickletools.genops(data):
                if opcode.name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE"}:
                    recent_string = argument
                    continue

                if recent_string != "imgsz":
                    continue

                if opcode.name in {"BININT", "BININT1", "BININT2"}:
                    return self._normalize_imgsz_to_dimensions(argument)
                if opcode.name == "NONE":
                    return None
        except Exception as exc:
            self.logger.warning("Failed to parse checkpoint pickle for %s: %s", source_path, exc)
        return None

    def _detect_model_inference_size_from_source_path(self, model_type, source_path):
        if model_type != "ul" or not source_path:
            return None
        if os.path.splitext(source_path)[1].lower() != ".pt":
            return None

        dimensions = self._detect_ultralytics_imgsz_with_torch(source_path)
        if dimensions is not None:
            return dimensions
        return self._detect_ultralytics_imgsz_from_zip(source_path)

    def _extract_rf_package_metadata_from_config(self, config):
        network_input = config["network_input"]
        assert isinstance(network_input, dict)

        size = network_input["training_input_size"]
        assert isinstance(size, dict)
        width = int(size["width"])
        height = int(size["height"])
        assert not (
            self._sanitize_inference_dimension(width) is None
            or self._sanitize_inference_dimension(height) is None
        ), "RF inference config training input size is out of range."

        preprocessing = {
            "training_input_size": {
                "width": width,
                "height": height,
            },
        }
        for key in (
            "color_mode",
            "resize_mode",
            "padding_value",
            "input_channels",
            "scaling_factor",
            "normalization",
            "dynamic_spatial_size_supported",
            "dynamic_spatial_size_mode",
        ):
            if key in network_input:
                preprocessing[key] = network_input[key]

        return {
            "inference_width": width,
            "inference_height": height,
            "preprocessing": preprocessing,
        }

    def _detect_rf_package_metadata_from_config_path(self, config_path):
        if not os.path.isfile(config_path):
            return {}

        with open(config_path, "r", encoding="utf-8") as config_input:
            config = json.load(config_input)

        metadata = self._extract_rf_package_metadata_from_config(config)
        metadata["inference_config_path"] = config_path
        return metadata

    def _detect_rf_package_metadata_from_source_path(self, source_path):
        base, ext = os.path.splitext(source_path or "")
        if ext.lower() != ".onnx":
            return {}

        config_path = f"{base}.inference_config.json"
        assert os.path.isfile(config_path), f"Missing RF package config: {config_path}"
        return self._detect_rf_package_metadata_from_config_path(config_path)

    def _detect_rf_package_metadata_from_engine_path(self, engine_path):
        if os.path.splitext(engine_path or "")[1].lower() != ".engine":
            return {}

        return self._detect_rf_package_metadata_from_config_path(f"{engine_path}.json")

    def _resolve_source_inference_size(self, model_type, model_name, *, source_paths=None):
        if model_type == "rf":
            source_candidates = tuple(source_paths or ()) or self._candidate_source_paths_for_model(model_type, model_name)
            for source_path in source_candidates:
                metadata = self._detect_rf_package_metadata_from_source_path(source_path)
                if metadata:
                    return {
                        "inference_width": metadata["inference_width"],
                        "inference_height": metadata["inference_height"],
                    }

            return dict(
                inference_width=self.DEFAULT_MODEL_INFERENCE_WIDTH,
                inference_height=self.DEFAULT_MODEL_INFERENCE_HEIGHT,
            )

        if model_type != "ul":
            return dict(
                inference_width=self.DEFAULT_MODEL_INFERENCE_WIDTH,
                inference_height=self.DEFAULT_MODEL_INFERENCE_HEIGHT,
            )

        source_candidates = tuple(source_paths or ()) or self._candidate_source_paths_for_model(model_type, model_name)
        for source_path in source_candidates:
            detected_dimensions = self._detect_model_inference_size_from_source_path(model_type, source_path)
            if detected_dimensions is not None:
                return detected_dimensions

        return dict(
            inference_width=self.DEFAULT_MODEL_INFERENCE_WIDTH,
            inference_height=self.DEFAULT_MODEL_INFERENCE_HEIGHT,
        )

    def _extract_inference_size_from_command(self, command):
        if not isinstance(command, (list, tuple)):
            return None

        imgsz_arg = next(
            (
                str(item).split("=", 1)[1]
                for item in command
                if isinstance(item, str) and item.startswith("imgsz=")
            ),
            None,
        )
        return self._normalize_imgsz_to_dimensions(imgsz_arg)

    def _inspect_engine_inference_size(self, engine_path):
        if not engine_path or not os.path.isfile(engine_path):
            return None

        try:
            import tensorrt as trt
        except Exception:
            return None

        try:
            logger = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(logger)
            with open(engine_path, "rb") as engine_input:
                engine = runtime.deserialize_cuda_engine(engine_input.read())
        except Exception as exc:
            self.logger.warning("Failed to inspect TensorRT engine %s: %s", engine_path, exc)
            return None

        if engine is None:
            return None

        try:
            for tensor_index in range(getattr(engine, "num_io_tensors", 0)):
                tensor_name = engine.get_tensor_name(tensor_index)
                if engine.get_tensor_mode(tensor_name) != trt.TensorIOMode.INPUT:
                    continue

                shape = tuple(engine.get_tensor_shape(tensor_name))
                dimensions = self._normalize_imgsz_to_dimensions(shape[-2:] if len(shape) >= 2 else None)
                if dimensions is not None:
                    return dimensions

                if -1 in shape and getattr(engine, "num_optimization_profiles", 0) > 0 and hasattr(engine, "get_tensor_profile_shape"):
                    min_shape, opt_shape, max_shape = engine.get_tensor_profile_shape(tensor_name, 0)
                    for candidate_shape in (opt_shape, max_shape, min_shape):
                        dimensions = self._normalize_imgsz_to_dimensions(
                            tuple(candidate_shape)[-2:] if len(candidate_shape) >= 2 else None
                        )
                        if dimensions is not None:
                            return dimensions
        except Exception as exc:
            self.logger.warning("Failed to read input tensor shape from %s: %s", engine_path, exc)
            return None

        return None

    def _is_tensorrt_engine_compatible(self, engine_path):
        if not engine_path or not os.path.isfile(engine_path):
            return False

        try:
            import tensorrt as trt

            logger = trt.Logger(trt.Logger.ERROR)
            runtime = trt.Runtime(logger)
            with open(engine_path, "rb") as engine_input:
                engine = runtime.deserialize_cuda_engine(engine_input.read())
            if engine is None:
                self.logger.warning(
                    "TensorRT rejected engine %s during compatibility check.",
                    engine_path,
                )
                return False
            context = engine.create_execution_context()
            if context is None:
                self.logger.warning(
                    "TensorRT could not create an execution context for %s.",
                    engine_path,
                )
                return False
            return True
        except Exception as exc:
            self.logger.warning(
                "TensorRT engine %s is not compatible with this machine: %s",
                engine_path,
                exc,
            )
            return False

    def _should_inspect_engine_inference_size(self, metadata):
        if not metadata:
            return False
        return metadata.get("tensorrt_version") == self.get_current_tensorrt_version()

    def _resolve_engine_inference_size(self, engine_path):
        metadata = self._read_compile_metadata(engine_path)
        if self._should_inspect_engine_inference_size(metadata):
            engine_dimensions = self._inspect_engine_inference_size(engine_path)
            if engine_dimensions is not None:
                return engine_dimensions

        direct_dimensions = self._normalize_imgsz_to_dimensions(
            [
                metadata.get("inference_height"),
                metadata.get("inference_width"),
            ]
            if metadata.get("inference_height") and metadata.get("inference_width")
            else None
        )
        if direct_dimensions is not None:
            return direct_dimensions

        command_dimensions = self._extract_inference_size_from_command(metadata.get("command"))
        if command_dimensions is not None:
            return command_dimensions

        return dict(
            inference_width=self.DEFAULT_MODEL_INFERENCE_WIDTH,
            inference_height=self.DEFAULT_MODEL_INFERENCE_HEIGHT,
        )

    def get_catalog_model_runtime_settings(self, model_type, model_name, *, source_paths=None, engine_path=None):
        settings = dict(
            inference_width=None,
            inference_height=None,
        )
        settings["default_confidence_threshold"] = self.get_catalog_model_default_confidence_threshold(
            model_type,
            model_name,
        )
        settings.update(
            self.get_catalog_model_tilletia_filter_settings(model_type, model_name)
        )

        dimensions = None
        if engine_path and model_type == "rf":
            package_metadata = self._detect_rf_package_metadata_from_engine_path(engine_path)
            if package_metadata:
                dimensions = {
                    "inference_width": package_metadata["inference_width"],
                    "inference_height": package_metadata["inference_height"],
                }
        elif engine_path and model_type == "ul":
            dimensions = self._resolve_engine_inference_size(engine_path)
        if dimensions is None:
            dimensions = self._resolve_source_inference_size(
                model_type,
                model_name,
                source_paths=source_paths,
            )

        settings["inference_width"] = self._sanitize_inference_dimension(dimensions.get("inference_width")) or self.DEFAULT_MODEL_INFERENCE_WIDTH
        settings["inference_height"] = self._sanitize_inference_dimension(dimensions.get("inference_height")) or self.DEFAULT_MODEL_INFERENCE_HEIGHT
        return settings

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

    def set_model_metadata(self, model_type, model_name, payload):
        entry = self._require_catalog_entry(model_type, model_name)
        payload = payload or {}

        with self.model_metadata_lock:
            metadata = self._read_model_metadata_unlocked()
            key = self._model_task_key(model_type, model_name)
            stored_entry = dict(metadata.get(key) or {})

            if "default_confidence_threshold" in payload:
                normalized_threshold = self._sanitize_default_confidence_threshold(
                    payload.get("default_confidence_threshold")
                )
                if normalized_threshold is None:
                    raise ModelValidationError("Invalid default confidence threshold.")
                if normalized_threshold == self.DEFAULT_MODEL_CONFIDENCE_THRESHOLD:
                    stored_entry.pop("default_confidence_threshold", None)
                else:
                    stored_entry["default_confidence_threshold"] = normalized_threshold

            supplied_filter_fields = [
                field
                for field in self.TILLETIA_FILTER_METADATA_FIELDS
                if field in payload
            ]
            if supplied_filter_fields:
                if len(supplied_filter_fields) != len(self.TILLETIA_FILTER_METADATA_FIELDS):
                    raise ModelValidationError("All Tilletia filter settings are required.")
                filter_values = {
                    field: self._sanitize_positive_integer(payload.get(field))
                    for field in self.TILLETIA_FILTER_METADATA_FIELDS
                }
                if any(value is None for value in filter_values.values()):
                    raise ModelValidationError("Invalid Tilletia filter settings.")
                stored_entry.update(filter_values)

            stored_entry = self._sanitize_model_metadata_entry(stored_entry)
            if stored_entry:
                metadata[key] = stored_entry
            else:
                metadata.pop(key, None)
            self._write_model_metadata_unlocked(metadata)

        saved_settings = self.get_catalog_model_runtime_settings(
            model_type,
            model_name,
            source_paths=entry.get("source_paths") or [],
            engine_path=(entry.get("engine") or {}).get("path"),
        )
        return dict(
            type=model_type,
            name=model_name,
            default_confidence_threshold=saved_settings["default_confidence_threshold"],
            tilletia_filter_max_width_px=saved_settings["tilletia_filter_max_width_px"],
            tilletia_filter_max_height_px=saved_settings["tilletia_filter_max_height_px"],
            tilletia_filter_training_width=saved_settings["tilletia_filter_training_width"],
            tilletia_filter_training_height=saved_settings["tilletia_filter_training_height"],
            inference_width=saved_settings["inference_width"],
            inference_height=saved_settings["inference_height"],
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

    def resolve_model_runtime_settings_for_path(self, model_path):
        defaults = self._default_model_runtime_settings()

        if not model_path:
            return defaults

        normalized = os.path.normpath(model_path)
        try:
            relative = os.path.relpath(normalized, self.model_dir)
        except ValueError:
            return defaults

        parts = relative.split(os.sep)
        if len(parts) < 2:
            return defaults

        model_type = parts[0]
        base_name = os.path.splitext(parts[-1])[0]
        source_candidates = self._candidate_source_paths_for_model(model_type, base_name)
        return self.get_catalog_model_runtime_settings(
            model_type,
            base_name,
            source_paths=source_candidates,
            engine_path=normalized if os.path.splitext(normalized)[1].lower() == ".engine" else None,
        )

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
        model_name=None,
        source_path=None,
        command=None,
        inference_width=None,
        inference_height=None,
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
        if model_name:
            payload["model_name"] = model_name
        if source_path:
            payload["source_path"] = source_path
        if command:
            payload["command"] = list(command)
        if inference_width is not None:
            payload["inference_width"] = int(inference_width)
        if inference_height is not None:
            payload["inference_height"] = int(inference_height)

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
            model_name = model.get("name")
            command = job.get("command")

        engine_path = self._resolve_compiled_engine_path(model_type, source_path)
        if not engine_path:
            return

        if model_type == "rf":
            source_base_path = os.path.splitext(source_path)[0]
            shutil.copy2(f"{source_base_path}.inference_config.json", f"{engine_path}.json")
            shutil.copy2(f"{source_base_path}.class_names.txt", f"{engine_path}.class_names.txt")
            self._append_compile_log(job_id, f"Saved RF runtime config: {engine_path}.json")
            self._append_compile_log(job_id, f"Saved RF class names: {engine_path}.class_names.txt")

        self._write_compile_metadata(
            engine_path,
            model_type=model_type,
            model_name=model_name,
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

    def _copy_zip_member(self, archive, member_name, target_path):
        with archive.open(member_name) as source, open(target_path, "wb") as target:
            shutil.copyfileobj(source, target)

    def _store_rf_package_upload(self, uploaded_file, target_dir, filename):
        package_name = os.path.splitext(filename)[0]
        temp_path = os.path.join(target_dir, f".{filename}.{uuid.uuid4().hex}.tmp")
        staged_paths = []
        temporary_member_paths = []

        try:
            uploaded_file.save(temp_path)
        except Exception as exc:
            self.logger.error("Failed to save uploaded RF package %s: %s", temp_path, exc)
            self._cleanup_partial_upload(temp_path)
            raise ModelManagerError("Failed to store uploaded model package.") from exc

        try:
            if not zipfile.is_zipfile(temp_path):
                raise ModelValidationError("RF model package must be a zip archive.")

            with zipfile.ZipFile(temp_path) as archive:
                archive_names = archive.namelist()
                member_names = set(archive_names)
                if len(member_names) != len(archive_names):
                    raise ModelValidationError("RF model package contains duplicate files.")

                required_members = [
                    f"{package_name}{suffix}"
                    for suffix in self.RF_PACKAGE_REQUIRED_SUFFIXES
                ]
                missing_members = [
                    member for member in required_members if member not in member_names
                ]
                if missing_members:
                    raise ModelValidationError(
                        "RF model package is missing required files: "
                        + ", ".join(missing_members)
                    )

                optional_members = [
                    f"{package_name}{suffix}"
                    for suffix in self.RF_PACKAGE_OPTIONAL_SUFFIXES
                    if f"{package_name}{suffix}" in member_names
                ]
                engine_members = [
                    f"{package_name}{suffix}"
                    for suffix in self.RF_PACKAGE_ENGINE_SUFFIXES
                ]
                present_engine_members = [
                    member for member in engine_members if member in member_names
                ]
                if present_engine_members and len(present_engine_members) != len(engine_members):
                    missing_engine_members = [
                        member for member in engine_members if member not in member_names
                    ]
                    raise ModelValidationError(
                        "RF model package has an incomplete TensorRT engine: "
                        + ", ".join(missing_engine_members)
                    )

                package_members = required_members + optional_members
                if present_engine_members:
                    package_members.extend(engine_members)

                unsafe_members = [
                    member
                    for member in archive_names
                    if not member
                    or member != os.path.basename(member)
                    or member in (".", "..")
                ]
                if unsafe_members:
                    raise ModelValidationError("RF model package contains unsafe file paths.")

                unexpected_members = sorted(member_names - set(package_members))
                if unexpected_members:
                    raise ModelValidationError(
                        "RF model package contains unexpected files: "
                        + ", ".join(unexpected_members)
                    )

                declared_target_paths = [
                    os.path.join(target_dir, member_name)
                    for member_name in package_members
                ]
                if any(os.path.exists(path) for path in declared_target_paths):
                    raise ModelConflictError("A model package with this name already exists.")

                for member_name in package_members:
                    temporary_member_path = os.path.join(
                        target_dir,
                        f".{member_name}.{uuid.uuid4().hex}.tmp",
                    )
                    self._copy_zip_member(archive, member_name, temporary_member_path)
                    temporary_member_paths.append(temporary_member_path)

                if present_engine_members:
                    temporary_engine_path = temporary_member_paths[
                        package_members.index(engine_members[0])
                    ]
                    if not self._is_tensorrt_engine_compatible(temporary_engine_path):
                        self.logger.warning(
                            "Ignoring incompatible TensorRT engine from RF package %s.",
                            filename,
                        )
                        retained_members = []
                        retained_temporary_paths = []
                        engine_member_set = set(engine_members)
                        for member_name, temporary_member_path in zip(
                            package_members, temporary_member_paths
                        ):
                            if member_name in engine_member_set:
                                self._cleanup_partial_upload(temporary_member_path)
                                continue
                            retained_members.append(member_name)
                            retained_temporary_paths.append(temporary_member_path)
                        package_members = retained_members
                        temporary_member_paths = retained_temporary_paths

                target_paths = [
                    os.path.join(target_dir, member_name)
                    for member_name in package_members
                ]
                if any(os.path.exists(path) for path in target_paths):
                    raise ModelConflictError("A model package with this name already exists.")

                for temporary_member_path, target_path in zip(
                    temporary_member_paths,
                    target_paths,
                ):
                    os.replace(temporary_member_path, target_path)
                    staged_paths.append(target_path)
        except (ModelValidationError, ModelConflictError):
            for path in temporary_member_paths:
                self._cleanup_partial_upload(path)
            for path in staged_paths:
                self._cleanup_partial_upload(path)
            raise
        except Exception as exc:
            self.logger.error("Failed to extract RF package %s: %s", temp_path, exc)
            for path in temporary_member_paths:
                self._cleanup_partial_upload(path)
            for path in staged_paths:
                self._cleanup_partial_upload(path)
            raise ModelManagerError("Failed to extract uploaded model package.") from exc
        finally:
            self._cleanup_partial_upload(temp_path)

        source_path = os.path.join(target_dir, f"{package_name}.onnx")
        try:
            metadata = self._detect_rf_package_metadata_from_source_path(source_path)
        except Exception:
            for path in staged_paths:
                self._cleanup_partial_upload(path)
            raise
        return package_name, source_path, metadata

    def _find_catalog_engine(self, model_type, model_name):
        base_dir = self._model_dir_for_type(model_type)
        candidate_paths = [os.path.join(base_dir, f"{model_name}.engine")]
        if not model_name.endswith("-fp16"):
            candidate_paths.append(os.path.join(base_dir, f"{model_name}-fp16.engine"))

        for candidate_path in candidate_paths:
            if not os.path.exists(candidate_path):
                continue
            if not self._is_tensorrt_engine_compatible(candidate_path):
                continue
            engine = self._new_engine_info(candidate_path)
            engine_dimensions = self._resolve_engine_inference_size(candidate_path)
            metadata = self._read_compile_metadata(candidate_path)
            if engine_dimensions is not None:
                engine.update(engine_dimensions)
            if metadata:
                engine.update(
                    dict(
                        tensorrt_version=metadata.get("tensorrt_version"),
                        compiled_at=metadata.get("compiled_at"),
                    )
                )
            return engine
        return None

    def _build_compile_command(self, model_type, source_path, *, inference_width=None, inference_height=None):
        if model_type == "ul":
            model_name = os.path.splitext(os.path.basename(source_path))[0]
            dimensions = self._normalize_imgsz_to_dimensions(
                [
                    inference_height,
                    inference_width,
                ]
                if inference_width is not None and inference_height is not None
                else None
            ) or self._resolve_source_inference_size(
                model_type,
                model_name,
                source_paths=(source_path,),
            )
            return [
                "yolo",
                "export",
                "format=engine",
                f"model={source_path}",
                f"imgsz={dimensions['inference_height']},{dimensions['inference_width']}",
                "half",
            ]
        return [sys.executable, self.convert_script, "--model", source_path]

    def _find_compile_source_path(self, entry):
        source_paths = tuple(entry.get("source_paths") or [])
        if entry.get("type") == "ul":
            extensions = (".pt",)
        elif entry.get("type") == "rf":
            extensions = (".pt", ".onnx")
        else:
            extensions = (".pt", ".onnx")

        for extension in extensions:
            source_path = next((path for path in source_paths if path.endswith(extension)), None)
            if source_path:
                return source_path
        return None

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
            for suffix in (
                ".pt",
                ".onnx",
                ".engine",
                *self.RF_PACKAGE_SIDECAR_SUFFIXES,
            )
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
        artifacts.update(
            runtime_path
            for runtime_path in (
                f"{path}{suffix}"
                for path in list(artifacts)
                if path.endswith(".engine")
                for suffix in (".json", ".class_names.txt")
            )
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

    def _ensure_rf_engine_output(self, job_id):
        with self.compile_jobs_lock:
            job = self.compile_jobs.get(job_id)
            if job is None:
                return False
            model = job.get("model") or {}
            source_path = model.get("source")

        expected_engine = self._resolve_compiled_engine_path("rf", source_path)
        if expected_engine and os.path.exists(expected_engine):
            self._append_compile_log(job_id, f"Engine ready: {expected_engine}")
            return True

        source_dir = os.path.dirname(source_path or "")
        source_base = os.path.splitext(os.path.basename(source_path or ""))[0]
        expected_engine = os.path.join(source_dir, f"{source_base}-fp16.engine")
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
            elif model_type == "rf":
                compile_artifacts_ready = self._ensure_rf_engine_output(job_id)
            if compile_artifacts_ready:
                try:
                    self._record_compile_metadata(job_id)
                except Exception as exc:
                    compile_artifacts_ready = False
                    self._append_compile_log(job_id, f"Failed to save compile metadata: {exc}")

        with self.compile_jobs_lock:
            job = self.compile_jobs.get(job_id)
            if job is not None:
                job["returncode"] = process.returncode
                job["finished_at"] = self._utcnow_text()
                job["status"] = "done" if compile_artifacts_ready else "error"

    def list_engine_models(self):
        models = []
        for entry in self.build_model_catalog():
            engine = entry.get("engine")
            if not entry.get("compiled") or not engine:
                continue
            models.append(
                dict(
                    path=engine.get("path"),
                    name=engine.get("name"),
                    type=entry.get("type"),
                    display=f"[{(entry.get('type') or '').upper()}] {engine.get('name')}",
                    default_confidence_threshold=entry.get("default_confidence_threshold", self.DEFAULT_MODEL_CONFIDENCE_THRESHOLD),
                    owner_username=entry.get("owner_username"),
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
                catalog_base_name = base_name
                if base_name.endswith("-fp16"):
                    source_base_name = base_name[:-5]
                    source_entry = model_map.get((model_type, source_base_name))
                    if source_entry and source_entry.get("source_paths"):
                        catalog_base_name = source_base_name
                existing_entry = model_map.get((model_type, catalog_base_name))
                if existing_entry and existing_entry.get("source_paths"):
                    continue
                if not self._is_tensorrt_engine_compatible(path):
                    continue
                self._get_or_create_catalog_entry(model_map, model_type, catalog_base_name)

        models = []
        for (model_type, base_name), entry in sorted(model_map.items(), key=lambda item: item[0]):
            engine = self._find_catalog_engine(model_type, base_name)
            source_settings = self.get_catalog_model_runtime_settings(
                model_type,
                base_name,
                source_paths=entry.get("source_paths") or [],
                engine_path=None,
            )
            runtime_settings = self.get_catalog_model_runtime_settings(
                model_type,
                base_name,
                source_paths=entry.get("source_paths") or [],
                engine_path=(engine or {}).get("path"),
            )
            entry["engine"] = engine
            entry["custom_input_size"] = False
            if model_type == "ul" and engine is not None:
                entry["custom_input_size"] = (
                    runtime_settings["inference_width"] != source_settings["inference_width"]
                    or runtime_settings["inference_height"] != source_settings["inference_height"]
                )
            entry["compiled"] = engine is not None
            entry["display"] = f"[{model_type.upper()}] {base_name}"
            entry["task"] = self.get_catalog_model_task(model_type, base_name)
            entry["default_confidence_threshold"] = runtime_settings["default_confidence_threshold"]
            entry["inference_width"] = runtime_settings["inference_width"]
            entry["inference_height"] = runtime_settings["inference_height"]
            entry["tilletia_filter_max_width_px"] = runtime_settings["tilletia_filter_max_width_px"]
            entry["tilletia_filter_max_height_px"] = runtime_settings["tilletia_filter_max_height_px"]
            entry["tilletia_filter_training_width"] = runtime_settings["tilletia_filter_training_width"]
            entry["tilletia_filter_training_height"] = runtime_settings["tilletia_filter_training_height"]
            if model_type == "rf":
                package_metadata = {}
                for source_path in entry.get("source_paths") or []:
                    package_metadata = self._detect_rf_package_metadata_from_source_path(source_path)
                    if package_metadata:
                        break
                if package_metadata:
                    entry["package_metadata"] = package_metadata
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
            raise ModelValidationError("Only .pt weights or RF .zip model packages are supported.")
        if ext == ".zip" and normalized_type != "rf":
            raise ModelValidationError("Zip model packages are only supported for RF models.")

        target_dir = self._model_dir_for_type(normalized_type)
        os.makedirs(target_dir, exist_ok=True)

        if ext == ".zip":
            model_name, target_path, package_metadata = self._store_rf_package_upload(
                uploaded_file,
                target_dir,
                filename,
            )
            self.auth.store_model_owner(normalized_type, model_name, owner_user_id)
            return dict(
                type=normalized_type,
                name=model_name,
                path=target_path,
                inference_width=None,
                inference_height=None,
                package_metadata=package_metadata,
            )

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
        detected_width = None
        detected_height = None
        if normalized_type == "ul":
            detected_dimensions = self._detect_model_inference_size_from_source_path(
                normalized_type,
                target_path,
            )
            detected_width = (detected_dimensions or {}).get("inference_width")
            detected_height = (detected_dimensions or {}).get("inference_height")

        return dict(
            type=normalized_type,
            name=model_name,
            path=target_path,
            inference_width=detected_width,
            inference_height=detected_height,
        )

    def start_compile(self, model_type, model_name, *, inference_width=None, inference_height=None):
        entry = self._require_catalog_entry(model_type, model_name)

        with self.compile_jobs_lock:
            existing_job = self._find_active_job_for_model(model_type, model_name)
            if existing_job is not None:
                return dict(job_id=existing_job["id"], already_running=True)

        source_path = self._find_compile_source_path(entry)
        if not source_path:
            raise ModelValidationError("No compilable source found for compilation.")

        command = self._build_compile_command(
            model_type,
            source_path,
            inference_width=inference_width,
            inference_height=inference_height,
        )
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
