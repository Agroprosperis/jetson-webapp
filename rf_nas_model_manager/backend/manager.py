import json
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile

from collections import deque
from datetime import datetime, timezone
from pathlib import Path


class ManagerError(Exception):
    pass


class ValidationError(ManagerError):
    pass


class ConflictError(ManagerError):
    pass


class NotFoundError(ManagerError):
    pass


class RoboflowError(ManagerError):
    pass


class RoboflowNotFoundError(RoboflowError):
    pass


class RoboflowClient:
    API_ROOT = "https://api.roboflow.com"

    def _get_payload(self, path, api_key):
        request = urllib.request.Request(
            f"{self.API_ROOT}{path}",
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise RoboflowNotFoundError("Roboflow resource was not found.") from exc
            if exc.code in (401, 403):
                raise RoboflowError("Roboflow rejected the API key.") from exc
            raise RoboflowError(f"Roboflow returned HTTP {exc.code}.") from exc
        except (urllib.error.URLError, TimeoutError) as exc:
            raise RoboflowError("Could not reach Roboflow.") from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RoboflowError("Roboflow returned an invalid response.") from exc
        return payload

    def _get(self, path, api_key):
        payload = self._get_payload(path, api_key)
        if not isinstance(payload, dict):
            raise RoboflowError("Roboflow returned an invalid response.")
        return payload

    def catalog(self, api_key):
        root = self._get("/", api_key)
        workspace_id = root.get("workspace")
        if not isinstance(workspace_id, str) or not workspace_id:
            raise RoboflowError("Roboflow did not return a workspace for this API key.")

        workspace_payload = self._get(
            f"/{urllib.parse.quote(workspace_id, safe='')}",
            api_key,
        )
        workspace = workspace_payload.get("workspace")
        if not isinstance(workspace, dict):
            raise RoboflowError("Roboflow did not return workspace details.")

        projects = []
        for project in workspace.get("projects", []):
            if not isinstance(project, dict):
                continue
            project_id = project.get("id")
            name = project.get("name")
            project_type = project.get("type")
            if not all(isinstance(value, str) and value for value in (project_id, name, project_type)):
                continue
            projects.append(
                {
                    "id": project_id,
                    "name": name,
                    "type": project_type,
                }
            )

        projects.sort(key=lambda item: (item["name"].lower(), item["id"]))
        workspace_name = workspace.get("name")
        return {
            "workspace": {
                "id": workspace_id,
                "name": workspace_name if isinstance(workspace_name, str) else workspace_id,
            },
            "projects": projects,
        }

    def models(self, api_key, project_id):
        parts = project_id.split("/")
        if len(parts) != 2 or not all(parts):
            raise ValidationError("project_id must use workspace/project format.")
        path = "/" + "/".join(urllib.parse.quote(part, safe="") for part in parts)
        payload = self._get(path, api_key)
        project = payload.get("project")
        if not isinstance(project, dict) or project.get("id") != project_id:
            raise RoboflowError("Roboflow returned a different project.")

        versions = payload.get("versions")
        if not isinstance(versions, list):
            raise RoboflowError("Roboflow did not return model versions for this project.")

        models = {}
        for version_payload in versions:
            if not isinstance(version_payload, dict):
                continue
            model = version_payload.get("model")
            if not isinstance(model, dict):
                continue
            model_id = model.get("id")
            version_id = version_payload.get("id")
            name = version_payload.get("name")
            created = version_payload.get("created")
            if not isinstance(model_id, str) or not re.fullmatch(r"[^/]+/[^/]+", model_id):
                continue
            if not isinstance(version_id, str) or not version_id:
                continue
            if not isinstance(name, str) or not name:
                name = model_id
            if isinstance(created, bool) or not isinstance(created, (int, float)):
                continue
            models[model_id] = {
                "id": model_id,
                "version": model_id.rsplit("/", 1)[1],
                "name": name,
                "created": created,
                "nas_group": None,
                "f1": None,
                "latency": None,
            }

        collection_query = urllib.parse.urlencode(
            {
                "status": "finished",
                "group": "false",
                "skipVersionModels": "true",
            }
        )
        collection_payload = self._get_payload(
            f"{path}/models?{collection_query}", api_key
        )
        if isinstance(collection_payload, dict):
            collection_models = collection_payload.get("models")
        else:
            collection_models = collection_payload
        if not isinstance(collection_models, list):
            raise RoboflowError("Roboflow returned an invalid project model catalog.")
        registry_models = [
            dict(model)
            for model in collection_models
            if isinstance(model, dict) and not model.get("group")
        ]

        for version_payload in versions:
            if not isinstance(version_payload, dict):
                continue
            version_id = version_payload.get("id")
            if not isinstance(version_id, str) or not version_id:
                continue
            version = version_id.rsplit("/", 1)[-1]
            encoded_version = urllib.parse.quote(version, safe="")
            training_path = f"{path}/{encoded_version}/training/results"
            try:
                training_results = self._get(training_path, api_key)
            except RoboflowNotFoundError:
                continue
            training_models = training_results.get("models")
            if not isinstance(training_models, list):
                continue
            mining_metrics = self._mining_metrics(training_results.get("mining"))
            nas_group = training_results.get("modelGroup")
            if not isinstance(nas_group, str) or not nas_group:
                nas_group = None
            detailed_models = training_models
            if nas_group:
                group_models = [
                    model
                    for model in collection_models
                    if isinstance(model, dict) and model.get("group") == nas_group
                ]
                if group_models:
                    detailed_models = group_models
            for training_model in detailed_models:
                if not isinstance(training_model, dict):
                    continue
                if training_model.get("nasFamily") == "baseline":
                    continue
                candidate = dict(training_model)
                candidate.setdefault("versionId", version)
                candidate.setdefault("created", version_payload.get("created"))
                candidate["nasGroup"] = nas_group
                candidate_id = (
                    candidate.get("modelId")
                    or candidate.get("url")
                    or candidate.get("fullUrl")
                )
                if isinstance(candidate_id, str):
                    mining_name = candidate_id.rsplit("-", 1)[-1]
                    candidate["miningMetrics"] = mining_metrics.get(mining_name)
                registry_models.append(candidate)

        for registry_model in registry_models:
            if not isinstance(registry_model, dict):
                continue
            model_id = registry_model.get("fullUrl")
            if not isinstance(model_id, str) or not model_id:
                model_id = registry_model.get("modelId") or registry_model.get("url")
                if isinstance(model_id, str) and "/" not in model_id:
                    model_id = f"{parts[0]}/{model_id}"
            if not isinstance(model_id, str) or not re.fullmatch(r"[^/]+/[^/]+", model_id):
                continue

            version = registry_model.get("version", registry_model.get("versionId"))
            if isinstance(version, bool) or not isinstance(version, (str, int, float)):
                version = ""
            else:
                version = str(version)
            name = (
                registry_model.get("versionName")
                or registry_model.get("modelDisplayName")
                or registry_model.get("displayName")
                or registry_model.get("name")
            )
            if not isinstance(name, str) or not name:
                name = model_id
            created_value = registry_model.get("created")
            if created_value is None:
                created_value = registry_model.get("createdAt")
            created = self._timestamp(created_value)
            nas_group = registry_model.get("nasGroup")
            if not isinstance(nas_group, str) or not nas_group:
                nas_group = None
            f1 = None
            latency = None
            mining_metric = registry_model.get("miningMetrics")
            if isinstance(mining_metric, dict):
                mining_f1 = mining_metric.get("f1")
                if isinstance(mining_f1, (int, float)) and not isinstance(mining_f1, bool):
                    f1 = mining_f1
            direct_f1 = registry_model.get("f1")
            if isinstance(direct_f1, (int, float)) and not isinstance(direct_f1, bool):
                f1 = direct_f1
            metrics = registry_model.get("metrics")
            if isinstance(metrics, dict):
                metric_f1 = metrics.get("f1")
                if isinstance(metric_f1, (int, float)) and not isinstance(metric_f1, bool):
                    f1 = metric_f1 / 100
                metric_latency = metrics.get("latency")
                if isinstance(metric_latency, (int, float)) and not isinstance(
                    metric_latency, bool
                ):
                    latency = metric_latency
            train_results = registry_model.get("trainResults")
            if not isinstance(train_results, dict):
                train = registry_model.get("train")
                train_results = train.get("results") if isinstance(train, dict) else None
            if isinstance(train_results, dict):
                result_f1 = train_results.get("f1")
                if isinstance(result_f1, (int, float)) and not isinstance(result_f1, bool):
                    f1 = result_f1
                result_latency = train_results.get("latency")
                if isinstance(result_latency, (int, float)) and not isinstance(
                    result_latency, bool
                ):
                    latency = result_latency
            models[model_id] = {
                "id": model_id,
                "version": version,
                "name": name,
                "created": created if created is not None else 0,
                "nas_group": nas_group,
                "f1": f1,
                "latency": latency,
            }

        result = sorted(
            models.values(),
            key=lambda item: item["created"],
            reverse=True,
        )
        return {"models": result}

    @staticmethod
    def _mining_metrics(value):
        metrics = {}

        def visit(item):
            if isinstance(item, dict):
                name = item.get("name")
                f1 = item.get("f1")
                if (
                    isinstance(name, str)
                    and name
                    and isinstance(f1, (int, float))
                    and not isinstance(f1, bool)
                ):
                    metrics[name] = {"f1": f1}
                for nested in item.values():
                    visit(nested)
            elif isinstance(item, list):
                for nested in item:
                    visit(nested)

        visit(value)
        return metrics

    @staticmethod
    def _timestamp(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return value / 1000 if value > 1e12 else value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
            except ValueError:
                return None
        if isinstance(value, dict):
            seconds = value.get("seconds", value.get("_seconds"))
            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool):
                return seconds
        return None


class ModelManager:
    REQUIRED_SOURCE_SUFFIXES = (
        ".onnx",
        ".inference_config.json",
        ".class_names.txt",
        ".model_config.json",
    )
    ENGINE_SUFFIXES = (
        "-fp16.engine",
        "-fp16.engine.json",
        "-fp16.engine.class_names.txt",
        "-fp16.engine.compile.json",
    )
    JOB_LOG_LIMIT = 500
    MODEL_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

    def __init__(self, data_dir="/data", roboflow_client=None):
        self.data_dir = Path(data_dir)
        self.models_dir = self.data_dir / "models"
        self.packages_dir = self.data_dir / "packages"
        self.work_dir = self.data_dir / "work"
        self.roboflow = roboflow_client or RoboflowClient()
        self.jobs = {}
        self.jobs_lock = threading.Lock()
        self.import_lock = threading.Lock()
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.packages_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _utcnow_text():
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def safe_model_name(model_id):
        name = model_id.strip().replace("/", "-")
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip(".-_")
        if not name:
            raise ValidationError("model_id does not produce a safe model name.")
        return name

    def catalog_from_roboflow(self, api_key):
        return self.roboflow.catalog(api_key)

    def models_from_roboflow(self, api_key, project_id):
        return self.roboflow.models(api_key, project_id)

    def _new_job(self, kind, model_name, model_id=None):
        job_id = uuid.uuid4().hex
        job = {
            "id": job_id,
            "kind": kind,
            "model_name": model_name,
            "model_id": model_id,
            "status": "queued",
            "created_at": self._utcnow_text(),
            "logs": deque(maxlen=self.JOB_LOG_LIMIT),
        }
        with self.jobs_lock:
            self.jobs[job_id] = job
        return job_id

    def _append_log(self, job_id, message):
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if job is not None:
                job["logs"].append(str(message))

    def _set_job(self, job_id, **updates):
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if job is not None:
                job.update(updates)

    def _serialize_job(self, job):
        return {
            "id": job["id"],
            "kind": job["kind"],
            "model_name": job["model_name"],
            "model_id": job.get("model_id"),
            "status": job["status"],
            "created_at": job["created_at"],
            "started_at": job.get("started_at"),
            "finished_at": job.get("finished_at"),
            "error": job.get("error"),
            "logs": list(job["logs"]),
        }

    def get_job(self, job_id):
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise NotFoundError("Job not found.")
            return self._serialize_job(job)

    def _active_job_for_model(self, model_name):
        with self.jobs_lock:
            return next(
                (
                    job
                    for job in self.jobs.values()
                    if job["model_name"] == model_name and job["status"] in ("queued", "running")
                ),
                None,
            )

    def start_import(self, model_id, api_key):
        if not self.MODEL_ID_PATTERN.fullmatch(model_id):
            raise ValidationError("model_id must use project/version format.")
        model_name = self.safe_model_name(model_id)
        if self._active_job_for_model(model_name):
            raise ConflictError("This model already has an active job.")
        if self._source_path(model_name, ".onnx").exists():
            raise ConflictError("This model is already imported.")

        job_id = self._new_job("import", model_name, model_id)
        thread = threading.Thread(
            target=self._run_import_job,
            args=(job_id, model_id, api_key),
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id}

    def _run_import_job(self, job_id, model_id, api_key):
        self._set_job(job_id, status="running", started_at=self._utcnow_text())
        try:
            with self.import_lock:
                self._import_model(job_id, model_id, api_key)
        except Exception as exc:
            error_message = str(exc).replace(api_key, "[redacted]")
            self._append_log(job_id, f"ERROR: {error_message}")
            self._set_job(
                job_id,
                status="error",
                error=error_message,
                finished_at=self._utcnow_text(),
            )
            return
        self._set_job(job_id, status="done", finished_at=self._utcnow_text())

    def _download_model(self, model_id, api_key, cache_dir):
        environment = os.environ.copy()
        environment.update(
            {
                "MODEL_CACHE_DIR": str(cache_dir),
                "TENSORRT_CACHE_PATH": str(cache_dir),
                "ONNXRUNTIME_EXECUTION_PROVIDERS": "[CPUExecutionProvider]",
            }
        )
        completed = subprocess.run(
            [
                "/opt/rf-inference/bin/python",
                "-m",
                "rf_nas_model_manager.backend.manager",
                "--download-worker",
            ],
            input=json.dumps({"model_id": model_id, "api_key": api_key}),
            text=True,
            capture_output=True,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            details = completed.stderr.strip() or completed.stdout.strip()
            message = details.splitlines()[-1] if details else "unknown downloader error"
            raise ManagerError(f"Roboflow download failed: {message}")

        prefix = "ROBOFLOW_MODEL_DIR="
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith(prefix):
                return Path(line[len(prefix):])
        raise ManagerError("Roboflow did not report the downloaded model directory.")

    @staticmethod
    def _read_json(path, label):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError(f"{label} is not valid JSON.") from exc
        if not isinstance(payload, dict):
            raise ValidationError(f"{label} must be a JSON object.")
        return payload

    def _validate_download(self, downloaded_dir):
        required = {
            "onnx": downloaded_dir / "weights.onnx",
            "inference_config": downloaded_dir / "inference_config.json",
            "class_names": downloaded_dir / "class_names.txt",
            "model_config": downloaded_dir / "model_config.json",
        }
        missing = [path.name for path in required.values() if not path.is_file()]
        if missing:
            raise ValidationError("Downloaded model is missing required files: " + ", ".join(missing))

        model_config = self._read_json(required["model_config"], "model_config.json")
        if model_config.get("model_architecture") != "rfdetr":
            raise ValidationError("This is not an RF-DETR NAS model.")
        if model_config.get("backend_type") != "onnx":
            raise ValidationError("RF-DETR model backend must be onnx.")
        if model_config.get("task_type") not in ("object-detection", "instance-segmentation"):
            raise ValidationError("RF-DETR model task is not supported.")

        inference_config = self._read_json(required["inference_config"], "inference_config.json")
        network_input = inference_config.get("network_input")
        size = network_input.get("training_input_size") if isinstance(network_input, dict) else None
        if not isinstance(size, dict):
            raise ValidationError("inference_config.json is missing the training input size.")
        for key in ("width", "height"):
            value = size.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 32 or value > 8192:
                raise ValidationError("inference_config.json has an invalid training input size.")

        class_names = [
            line.strip()
            for line in required["class_names"].read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not class_names:
            raise ValidationError("class_names.txt is empty.")
        return required

    def _import_model(self, job_id, model_id, api_key):
        model_name = self.safe_model_name(model_id)
        job_dir = self.work_dir / job_id
        cache_dir = job_dir / "cache"
        stage_dir = job_dir / "stage"
        cache_dir.mkdir(parents=True, exist_ok=True)
        stage_dir.mkdir(parents=True, exist_ok=True)
        self._append_log(job_id, f"Downloading {model_id}...")

        try:
            downloaded_dir = self._download_model(model_id, api_key, cache_dir)
            required = self._validate_download(downloaded_dir)
            self._append_log(job_id, "Confirmed RF-DETR NAS ONNX package.")

            staged = {
                stage_dir / f"{model_name}.onnx": required["onnx"],
                stage_dir / f"{model_name}.inference_config.json": required["inference_config"],
                stage_dir / f"{model_name}.class_names.txt": required["class_names"],
                stage_dir / f"{model_name}.model_config.json": required["model_config"],
            }
            for target, source in staged.items():
                shutil.copy2(source, target)

            final_paths = [self.models_dir / path.name for path in staged]
            if any(path.exists() for path in final_paths):
                raise ConflictError("This model is already imported.")

            moved = []
            try:
                for source, target in zip(staged, final_paths):
                    os.replace(source, target)
                    moved.append(target)
                self._write_manifest(model_name, model_id)
            except Exception:
                for path in moved:
                    path.unlink(missing_ok=True)
                self._source_path(model_name, ".manifest.json").unlink(missing_ok=True)
                raise
            self._append_log(job_id, f"Imported model as {model_name}.")
        finally:
            shutil.rmtree(job_dir, ignore_errors=True)

    def _source_path(self, model_name, suffix):
        return self.models_dir / f"{model_name}{suffix}"

    def _engine_path(self, model_name):
        return self.models_dir / f"{model_name}-fp16.engine"

    def _artifact_paths(self, model_name):
        paths = [self._source_path(model_name, suffix) for suffix in self.REQUIRED_SOURCE_SUFFIXES]
        paths.append(self._source_path(model_name, ".manifest.json"))
        paths.extend(self._source_path(model_name, suffix) for suffix in self.ENGINE_SUFFIXES)
        return paths

    def _require_model(self, model_name):
        if model_name != self.safe_model_name(model_name):
            raise ValidationError("Invalid model name.")
        if not self._source_path(model_name, ".onnx").is_file():
            raise NotFoundError("Model not found.")

    def _manifest_payload(self, model_name, model_id):
        artifacts = [
            path.name
            for path in self._artifact_paths(model_name)
            if path.is_file() and not path.name.endswith(".manifest.json")
        ]
        return {
            "model_id": model_id,
            "resolved_model_id": model_id,
            "artifacts": artifacts,
            "zip": f"{model_name}.zip",
        }

    def _write_manifest(self, model_name, model_id):
        path = self._source_path(model_name, ".manifest.json")
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        temporary.write_text(
            json.dumps(self._manifest_payload(model_name, model_id), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

    def _metadata_for_model(self, model_name):
        config = self._read_json(
            self._source_path(model_name, ".inference_config.json"),
            "inference_config.json",
        )
        network_input = config["network_input"]
        size = network_input["training_input_size"]
        manifest = self._read_json(
            self._source_path(model_name, ".manifest.json"),
            "manifest.json",
        )
        engine_path = self._engine_path(model_name)
        engine = None
        if engine_path.is_file():
            compile_metadata = self._read_json(
                self._source_path(model_name, "-fp16.engine.compile.json"),
                "compile metadata",
            )
            engine = {
                "name": engine_path.name,
                "tensorrt_version": compile_metadata.get("tensorrt_version"),
                "compiled_at": compile_metadata.get("compiled_at"),
            }
        return {
            "name": model_name,
            "model_id": manifest.get("model_id"),
            "inference_width": size["width"],
            "inference_height": size["height"],
            "preprocessing": {
                key: network_input.get(key)
                for key in ("color_mode", "resize_mode", "scaling_factor", "normalization")
            },
            "compiled": engine is not None,
            "engine": engine,
        }

    def list_models(self):
        models = []
        for source_path in sorted(self.models_dir.glob("*.onnx")):
            try:
                models.append(self._metadata_for_model(source_path.stem))
            except (KeyError, ManagerError):
                continue
        return {"models": models, "tensorrt": self.tensorrt_capabilities()}

    @staticmethod
    def tensorrt_capabilities():
        try:
            import tensorrt as trt
        except Exception:
            return {"available": False, "version": None}
        try:
            import torch
        except Exception:
            return {"available": False, "version": trt.__version__}
        return {"available": bool(torch.cuda.is_available()), "version": trt.__version__}

    def start_compile(self, model_name):
        self._require_model(model_name)
        if self._active_job_for_model(model_name):
            raise ConflictError("This model already has an active job.")
        if not self.tensorrt_capabilities()["available"]:
            raise ValidationError("TensorRT is not available.")
        job_id = self._new_job("compile", model_name)
        thread = threading.Thread(target=self._run_compile_job, args=(job_id, model_name), daemon=True)
        thread.start()
        return {"job_id": job_id}

    def _run_compile_job(self, job_id, model_name):
        self._set_job(job_id, status="running", started_at=self._utcnow_text())
        try:
            self._compile_model(job_id, model_name)
        except Exception as exc:
            self._append_log(job_id, f"ERROR: {exc}")
            self._set_job(
                job_id,
                status="error",
                error=str(exc),
                finished_at=self._utcnow_text(),
            )
            return
        self._set_job(job_id, status="done", finished_at=self._utcnow_text())

    def _compile_model(self, job_id, model_name):
        import tensorrt as trt

        onnx_path = self._source_path(model_name, ".onnx")
        engine_path = self._engine_path(model_name)
        compile_root = self.work_dir / job_id
        compile_dir = compile_root / "compile"
        compile_dir.mkdir(parents=True, exist_ok=True)

        try:
            temporary_engine = compile_dir / engine_path.name
            temporary_config = compile_dir / f"{engine_path.name}.json"
            temporary_classes = compile_dir / f"{engine_path.name}.class_names.txt"
            temporary_metadata = compile_dir / f"{engine_path.name}.compile.json"
            self._append_log(job_id, f"TensorRT version: {trt.__version__}")
            self._append_log(job_id, f"Parsing {onnx_path.name}...")

            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            parser = trt.OnnxParser(network, logger)
            config = builder.create_builder_config()
            config.set_memory_pool_limit(
                trt.MemoryPoolType.WORKSPACE,
                2 * 1024 * 1024 * 1024,
            )
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                self._append_log(job_id, "FP16 enabled.")

            if not parser.parse(onnx_path.read_bytes()):
                errors = [
                    str(parser.get_error(index))
                    for index in range(parser.num_errors)
                ]
                raise ManagerError(
                    "TensorRT could not parse the ONNX model: " + "; ".join(errors)
                )

            self._append_log(job_id, "Building TensorRT engine...")
            serialized = builder.build_serialized_network(network, config)
            if serialized is None:
                raise ManagerError("TensorRT engine build failed.")
            temporary_engine.write_bytes(serialized)
            shutil.copy2(
                self._source_path(model_name, ".inference_config.json"),
                temporary_config,
            )
            shutil.copy2(
                self._source_path(model_name, ".class_names.txt"),
                temporary_classes,
            )
            temporary_metadata.write_text(
                json.dumps(
                    {
                        "tensorrt_version": trt.__version__,
                        "compiled_at": self._utcnow_text(),
                        "model_type": "rf",
                        "model_name": model_name,
                        "source_path": onnx_path.name,
                    },
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )

            replacements = (
                (
                    temporary_config,
                    self._source_path(model_name, "-fp16.engine.json"),
                ),
                (
                    temporary_classes,
                    self._source_path(model_name, "-fp16.engine.class_names.txt"),
                ),
                (
                    temporary_metadata,
                    self._source_path(model_name, "-fp16.engine.compile.json"),
                ),
                (temporary_engine, engine_path),
            )
            for source, target in replacements:
                os.replace(source, target)

            manifest = self._read_json(
                self._source_path(model_name, ".manifest.json"),
                "manifest.json",
            )
            self._write_manifest(model_name, manifest["model_id"])
            self._append_log(job_id, f"Engine ready: {engine_path.name}")
        finally:
            shutil.rmtree(compile_root, ignore_errors=True)

    def create_package(self, model_name):
        self._require_model(model_name)
        if self._active_job_for_model(model_name):
            raise ConflictError("Cannot package a model with an active job.")
        manifest = self._read_json(self._source_path(model_name, ".manifest.json"), "manifest.json")
        self._write_manifest(model_name, manifest["model_id"])
        package_path = self.packages_dir / f"{model_name}.zip"
        temporary = package_path.with_name(f".{package_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for path in self._artifact_paths(model_name):
                    if path.is_file():
                        archive.write(path, arcname=path.name)
            os.replace(temporary, package_path)
        finally:
            temporary.unlink(missing_ok=True)
        return package_path

    def delete_model(self, model_name):
        self._require_model(model_name)
        if self._active_job_for_model(model_name):
            raise ConflictError("Cannot delete a model with an active job.")
        deleted = []
        for path in self._artifact_paths(model_name) + [self.packages_dir / f"{model_name}.zip"]:
            if path.is_file():
                path.unlink()
                deleted.append(path.name)
        return {"name": model_name, "deleted": deleted}


def _run_download_worker():
    payload = json.load(sys.stdin)
    model_id = payload.get("model_id")
    api_key = payload.get("api_key")
    if not isinstance(model_id, str) or not isinstance(api_key, str):
        raise ValueError("Invalid downloader request.")

    cache_dir = Path(os.environ["MODEL_CACHE_DIR"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["ROBOFLOW_API_KEY"] = api_key

    from inference import get_model
    from PIL import Image

    loaded_model_dir = {}

    def point_model_directory(model_dir):
        loaded_model_dir["path"] = model_dir

    model = get_model(
        model_id=model_id,
        api_key=api_key,
        point_model_directory=point_model_directory,
    )
    model.infer(Image.new("RGB", (960, 960), color=(0, 0, 0)))

    model_dir = loaded_model_dir.get("path")
    if not model_dir:
        raise RuntimeError("Roboflow did not report the downloaded model directory.")
    print(f"ROBOFLOW_MODEL_DIR={model_dir}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] != ["--download-worker"]:
        raise SystemExit("Unsupported manager command.")
    _run_download_worker()
