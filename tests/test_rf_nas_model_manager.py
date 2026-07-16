import io
import json
import sys
import tempfile
import unittest
import zipfile

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
for path in (ROOT_DIR, SRC_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_manager import ModelManager as TilletiaModelManager, ModelValidationError
from rf_nas_model_manager.backend.app import create_app
from rf_nas_model_manager.backend.manager import (
    ModelManager,
    RoboflowClient,
    RoboflowNotFoundError,
    ValidationError,
)


class FakeAuth:
    def get_model_owner_username(self, model_type, model_name):
        return "tester"

    def store_model_owner(self, model_type, model_name, owner_user_id):
        return None

    def delete_model_owner(self, model_type, model_name):
        return None


class FakeUpload:
    def __init__(self, filename, content):
        self.filename = filename
        self.content = content

    def save(self, path):
        Path(path).write_bytes(self.content)


class StubApiManager:
    def __init__(self):
        self.calls = []

    def catalog_from_roboflow(self, api_key):
        self.calls.append(("catalog", api_key))
        return {
            "workspace": {"id": "workspace", "name": "Workspace"},
            "projects": [],
        }

    def models_from_roboflow(self, api_key, project_id):
        self.calls.append(("models", api_key, project_id))
        return {"models": []}

    def start_import(self, model_id, api_key):
        self.calls.append(("import", model_id, api_key))
        return {"job_id": "job-1"}

    def list_models(self):
        return {"models": [], "tensorrt": {"available": False, "version": None}}

    def start_compile(self, model_name):
        return {"job_id": "compile-1"}

    def get_job(self, job_id):
        return {"id": job_id, "status": "done", "logs": []}

    def create_package(self, model_name):
        raise AssertionError("not used")

    def delete_model(self, model_name):
        return {"name": model_name, "deleted": []}


class StubRoboflowClient(RoboflowClient):
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def _get(self, path, api_key):
        self.requests.append((path, api_key))
        response = self.responses[path]
        if isinstance(response, Exception):
            raise response
        return response

    def _get_payload(self, path, api_key):
        self.requests.append((path, api_key))
        response = self.responses[path]
        if isinstance(response, Exception):
            raise response
        return response


def inference_config():
    return {
        "network_input": {
            "training_input_size": {"width": 800, "height": 800},
            "color_mode": "rgb",
            "resize_mode": "stretch",
            "input_channels": 3,
            "scaling_factor": 255,
            "normalization": [
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225],
            ],
        }
    }


class StrictApiContractTests(unittest.TestCase):
    def setUp(self):
        self.manager = StubApiManager()
        self.client = create_app(self.manager).test_client()

    def test_catalog_accepts_only_exact_contract(self):
        response = self.client.post("/api/roboflow/catalog", json={"api_key": "secret"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.manager.calls, [("catalog", "secret")])
        self.assertNotIn("secret", response.get_data(as_text=True))

        for payload in ({}, {"api_key": "secret", "extra": "value"}, {"api_key": 7}):
            response = self.client.post("/api/roboflow/catalog", json=payload)
            self.assertEqual(response.status_code, 400)

    def test_models_and_import_accept_only_exact_contracts(self):
        response = self.client.post(
            "/api/roboflow/models",
            json={"api_key": "secret", "project_id": "workspace/project"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.manager.calls[-1], ("models", "secret", "workspace/project"))

        response = self.client.post(
            "/api/models/import",
            json={"api_key": "secret", "model_id": "project/1"},
        )
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.manager.calls[-1], ("import", "project/1", "secret"))

        response = self.client.post(
            "/api/models/import",
            json={"api_key": "secret", "model_id": "project/1", "extra": "value"},
        )
        self.assertEqual(response.status_code, 400)


class RoboflowDiscoveryTests(unittest.TestCase):
    def test_lists_workspace_projects_and_trained_models(self):
        client = StubRoboflowClient(
            {
                "/": {"workspace": "workspace"},
                "/workspace": {
                    "workspace": {
                        "name": "My Workspace",
                        "projects": [
                            {"id": "workspace/b", "name": "B", "type": "object-detection"},
                            {"id": "workspace/a", "name": "A", "type": "instance-segmentation"},
                        ],
                    }
                },
                "/workspace/a": {
                    "project": {
                        "id": "workspace/a",
                        "versions": 3,
                    },
                    "versions": [
                            {
                                "id": "workspace/a/51",
                                "name": "NAS run",
                                "created": 50.0,
                            },
                            {
                                "id": "workspace/a/2",
                                "name": "trained",
                                "created": 20.0,
                                "model": {"id": "a/2"},
                            },
                            {
                                "id": "workspace/a/1",
                                "name": "not trained",
                                "created": 10.0,
                            },
                    ],
                },
                "/workspace/a/models?status=finished&group=false&skipVersionModels=true": {
                    "models": [
                        {
                            "modelId": "a-instant-1",
                            "versionId": 1,
                            "modelDisplayName": "Roboflow Instant",
                            "createdAt": "1970-01-01T00:00:30Z",
                        },
                        {
                            "modelId": "a-51-baseline",
                            "group": "nas-51",
                            "nasFamily": "baseline",
                            "metrics": {"latency": 9.75},
                        },
                        {
                            "modelId": "a-51-nas-gpu-fast",
                            "group": "nas-51",
                            "modelDisplayName": "NAS fast",
                            "nasFamily": "small",
                            "metrics": {"latency": 4.849664211273193},
                        },
                    ]
                },
                "/workspace/a/51/training/results": {
                    "status": "finished",
                    "modelGroup": "nas-51",
                    "modelCount": 1,
                    "mining": {
                        "snapshots": [
                            {"models": [{"name": "fast", "f1": 0.966}]}
                        ]
                    },
                    "models": [
                        {
                            "modelId": "a-51-nas-gpu-fast",
                        },
                    ],
                },
                "/workspace/a/2/training/results": RoboflowNotFoundError("not found"),
                "/workspace/a/1/training/results": RoboflowNotFoundError("not found"),
            }
        )

        catalog = client.catalog("secret")
        models = client.models("secret", "workspace/a")

        self.assertEqual([item["id"] for item in catalog["projects"]], ["workspace/a", "workspace/b"])
        self.assertEqual(
            models["models"],
            [
                {
                    "id": "workspace/a-51-nas-gpu-fast",
                    "version": "51",
                    "name": "NAS fast",
                    "created": 50.0,
                    "nas_group": "nas-51",
                    "f1": 0.966,
                    "latency": 4.849664211273193,
                },
                {
                    "id": "workspace/a-instant-1",
                    "version": "1",
                    "name": "Roboflow Instant",
                    "created": 30.0,
                    "nas_group": None,
                    "f1": None,
                    "latency": None,
                },
                {
                    "id": "a/2",
                    "version": "2",
                    "name": "trained",
                    "created": 20.0,
                    "nas_group": None,
                    "f1": None,
                    "latency": None,
                },
            ],
        )
        self.assertIn(
            (
                "/workspace/a/models?status=finished&group=false&skipVersionModels=true",
                "secret",
            ),
            client.requests,
        )
        self.assertNotIn("secret", json.dumps(catalog))
        self.assertNotIn("secret", json.dumps(models))


class PackageProcessingTests(unittest.TestCase):
    def test_download_uses_isolated_inference_worker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ModelManager(data_dir=Path(temp_dir) / "data")
            completed = SimpleNamespace(
                returncode=0,
                stdout="inference output\nROBOFLOW_MODEL_DIR=/tmp/model\n",
                stderr="",
            )
            with patch(
                "rf_nas_model_manager.backend.manager.subprocess.run",
                return_value=completed,
            ) as run:
                model_dir = manager._download_model(
                    "project/model", "secret", Path(temp_dir) / "cache"
                )

            command = run.call_args.args[0]
            request = json.loads(run.call_args.kwargs["input"])
            self.assertEqual(command[0], "/opt/rf-inference/bin/python")
            self.assertNotIn("secret", command)
            self.assertEqual(request, {"model_id": "project/model", "api_key": "secret"})
            self.assertEqual(model_dir, Path("/tmp/model"))

    def make_downloaded_model(self, root, architecture="rfdetr"):
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        (root / "weights.onnx").write_bytes(b"onnx")
        (root / "inference_config.json").write_text(json.dumps(inference_config()), encoding="utf-8")
        (root / "class_names.txt").write_text("Tilletia\n", encoding="utf-8")
        (root / "model_config.json").write_text(
            json.dumps(
                {
                    "model_architecture": architecture,
                    "task_type": "object-detection",
                    "backend_type": "onnx",
                }
            ),
            encoding="utf-8",
        )
        return root

    def test_rejects_non_rfdetr_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = ModelManager(data_dir=Path(temp_dir) / "data")
            downloaded = self.make_downloaded_model(Path(temp_dir) / "download", architecture="yolov8")
            with self.assertRaisesRegex(ValidationError, "not an RF-DETR NAS model"):
                manager._validate_download(downloaded)

    def test_compiled_package_imports_into_tilletia(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            standalone = ModelManager(data_dir=root / "standalone")
            name = "workspace-project-1"
            model_id = "project/1"

            (standalone.models_dir / f"{name}.onnx").write_bytes(b"onnx")
            (standalone.models_dir / f"{name}.inference_config.json").write_text(
                json.dumps(inference_config()), encoding="utf-8"
            )
            (standalone.models_dir / f"{name}.class_names.txt").write_text("Tilletia\n", encoding="utf-8")
            (standalone.models_dir / f"{name}.model_config.json").write_text(
                json.dumps(
                    {
                        "model_architecture": "rfdetr",
                        "task_type": "object-detection",
                        "backend_type": "onnx",
                    }
                ),
                encoding="utf-8",
            )
            (standalone.models_dir / f"{name}-fp16.engine").write_bytes(b"engine")
            (standalone.models_dir / f"{name}-fp16.engine.json").write_text(
                json.dumps(inference_config()), encoding="utf-8"
            )
            (standalone.models_dir / f"{name}-fp16.engine.class_names.txt").write_text(
                "Tilletia\n", encoding="utf-8"
            )
            (standalone.models_dir / f"{name}-fp16.engine.compile.json").write_text(
                json.dumps({"tensorrt_version": "10.15", "compiled_at": "now"}),
                encoding="utf-8",
            )
            standalone._write_manifest(name, model_id)
            package = standalone.create_package(name)

            tilletia = TilletiaModelManager(auth_module=FakeAuth(), model_dir=root / "tilletia")
            tilletia._is_tensorrt_engine_compatible = lambda engine_path: True
            uploaded = tilletia.upload_model(
                "rf",
                FakeUpload(package.name, package.read_bytes()),
                owner_user_id=1,
            )
            catalog = tilletia.build_model_catalog()

            self.assertEqual(uploaded["name"], name)
            self.assertTrue(catalog[0]["compiled"])
            self.assertEqual(catalog[0]["engine"]["name"], f"{name}-fp16.engine")

    def test_tilletia_discards_incompatible_engine_but_keeps_raw_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TilletiaModelManager(auth_module=FakeAuth(), model_dir=temp_dir)
            manager._is_tensorrt_engine_compatible = lambda engine_path: False
            name = "sample"

            package = io.BytesIO()
            with zipfile.ZipFile(package, "w") as archive:
                archive.writestr(f"{name}.onnx", b"onnx")
                archive.writestr(f"{name}.inference_config.json", json.dumps(inference_config()))
                archive.writestr(f"{name}.class_names.txt", "Tilletia\n")
                archive.writestr(f"{name}-fp16.engine", b"incompatible")
                archive.writestr(f"{name}-fp16.engine.json", json.dumps(inference_config()))
                archive.writestr(f"{name}-fp16.engine.class_names.txt", "Tilletia\n")
                archive.writestr(
                    f"{name}-fp16.engine.compile.json",
                    json.dumps({"tensorrt_version": "different-machine"}),
                )

            manager.upload_model("rf", FakeUpload(f"{name}.zip", package.getvalue()), 1)

            model_dir = Path(temp_dir, "rf")
            self.assertTrue((model_dir / f"{name}.onnx").is_file())
            self.assertTrue((model_dir / f"{name}.inference_config.json").is_file())
            self.assertTrue((model_dir / f"{name}.class_names.txt").is_file())
            self.assertFalse((model_dir / f"{name}-fp16.engine").exists())
            self.assertFalse((model_dir / f"{name}-fp16.engine.json").exists())
            self.assertFalse((model_dir / f"{name}-fp16.engine.class_names.txt").exists())
            self.assertFalse((model_dir / f"{name}-fp16.engine.compile.json").exists())

            catalog = manager.build_model_catalog()
            self.assertEqual(len(catalog), 1)
            self.assertFalse(catalog[0]["compiled"])
            self.assertIsNone(catalog[0]["engine"])

    def test_tilletia_rejects_partial_or_unsafe_compiled_package(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = TilletiaModelManager(auth_module=FakeAuth(), model_dir=temp_dir)
            name = "sample"

            partial = io.BytesIO()
            with zipfile.ZipFile(partial, "w") as archive:
                archive.writestr(f"{name}.onnx", b"onnx")
                archive.writestr(f"{name}.inference_config.json", json.dumps(inference_config()))
                archive.writestr(f"{name}.class_names.txt", "Tilletia\n")
                archive.writestr(f"{name}-fp16.engine", b"engine")
            with self.assertRaisesRegex(ModelValidationError, "incomplete TensorRT engine"):
                manager.upload_model("rf", FakeUpload(f"{name}.zip", partial.getvalue()), 1)

            unsafe = io.BytesIO()
            with zipfile.ZipFile(unsafe, "w") as archive:
                archive.writestr(f"{name}.onnx", b"onnx")
                archive.writestr(f"{name}.inference_config.json", json.dumps(inference_config()))
                archive.writestr(f"{name}.class_names.txt", "Tilletia\n")
                archive.writestr("nested/unexpected.txt", "bad")
            with self.assertRaisesRegex(ModelValidationError, "unsafe file paths"):
                manager.upload_model("rf", FakeUpload(f"{name}.zip", unsafe.getvalue()), 1)


if __name__ == "__main__":
    unittest.main()
