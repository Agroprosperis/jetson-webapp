import json
import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_manager import ModelManager, ModelValidationError


class FakeAuth:
    def get_model_owner_username(self, model_type, model_name):
        return "tester"

    def store_model_owner(self, model_type, model_name, owner_user_id):
        return None

    def delete_model_owner(self, model_type, model_name):
        return None


class FakeUpload:
    def __init__(self, filename, content=b"weights"):
        self.filename = filename
        self._content = content

    def save(self, target_path):
        Path(target_path).write_bytes(self._content)


class TestableModelManager(ModelManager):
    def __init__(self, *args, detected_dimensions=None, engine_dimensions=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.detected_dimensions = detected_dimensions
        self.engine_dimensions = engine_dimensions or {}

    def _detect_model_inference_size_from_source_path(self, model_type, source_path):
        return self.detected_dimensions

    def _inspect_engine_inference_size(self, engine_path):
        return self.engine_dimensions.get(str(engine_path))


class ModelManagerRuntimeSettingsTests(unittest.TestCase):
    def make_manager(self, temp_dir, *, detected_dimensions=None, engine_dimensions=None):
        return TestableModelManager(
            auth_module=FakeAuth(),
            model_dir=temp_dir,
            compile_cwd=temp_dir,
            convert_script=str(Path(temp_dir) / "convert.py"),
            detected_dimensions=detected_dimensions,
            engine_dimensions=engine_dimensions,
        )

    def touch_model_source(self, temp_dir, relative_path):
        path = Path(temp_dir, relative_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
        return path

    def make_rf_package_zip(self, package_name):
        buffer = io.BytesIO()
        inference_config = {
            "network_input": {
                "training_input_size": {
                    "width": 960,
                    "height": 960,
                },
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
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(f"{package_name}.onnx", b"onnx")
            archive.writestr(
                f"{package_name}.inference_config.json",
                json.dumps(inference_config),
            )
            archive.writestr(f"{package_name}.model_config.json", "{}")
            archive.writestr(f"{package_name}.class_names.txt", "tilletia\n")
        return buffer.getvalue()

    def test_upload_model_returns_autodetected_source_size_without_persisting_dimensions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(
                temp_dir,
                detected_dimensions={
                    "inference_width": 1280,
                    "inference_height": 736,
                },
            )

            uploaded = manager.upload_model("ul", FakeUpload("sample.pt"), owner_user_id=7)

            self.assertEqual(uploaded["inference_width"], 1280)
            self.assertEqual(uploaded["inference_height"], 736)

            metadata_path = Path(temp_dir, "model_metadata.json")
            self.assertFalse(metadata_path.exists())

            catalog = manager.build_model_catalog()
            self.assertEqual(len(catalog), 1)
            self.assertEqual(catalog[0]["inference_width"], 1280)
            self.assertEqual(catalog[0]["inference_height"], 736)
            self.assertNotIn("sliding_window", catalog[0])

    def test_compile_command_uses_requested_compile_size_and_metadata_only_keeps_default_confidence(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(temp_dir)
            source_path = self.touch_model_source(temp_dir, "ul/sample.pt")

            saved = manager.set_model_metadata(
                "ul",
                "sample",
                {
                    "default_confidence_threshold": 0.55,
                },
            )

            self.assertEqual(saved["default_confidence_threshold"], 0.55)
            self.assertEqual(saved["inference_width"], 640)
            self.assertEqual(saved["inference_height"], 640)

            metadata = json.loads(Path(temp_dir, "model_metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(
                metadata["ul:sample"],
                {
                    "default_confidence_threshold": 0.55,
                },
            )

            command = manager._build_compile_command(
                "ul",
                str(source_path),
                inference_width=1024,
                inference_height=768,
            )
            self.assertIn("imgsz=768,1024", command)

    def test_custom_input_size_uses_engine_shape_for_catalog_and_runtime(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            engine_path = Path(temp_dir, "ul/sample.engine")
            manager = self.make_manager(
                temp_dir,
                detected_dimensions={
                    "inference_width": 640,
                    "inference_height": 640,
                },
                engine_dimensions={
                    str(engine_path): {
                        "inference_width": 1024,
                        "inference_height": 768,
                    },
                },
            )
            self.touch_model_source(temp_dir, "ul/sample.pt")
            self.touch_model_source(temp_dir, "ul/sample.engine")
            manager._write_compile_metadata(
                str(engine_path),
                model_type="ul",
                model_name="sample",
                command=["yolo", "export", "imgsz=640,640"],
            )

            catalog = manager.build_model_catalog()
            self.assertEqual(len(catalog), 1)
            self.assertTrue(catalog[0]["custom_input_size"])
            self.assertTrue(catalog[0]["compiled"])
            self.assertEqual(catalog[0]["inference_width"], 1024)
            self.assertEqual(catalog[0]["inference_height"], 768)
            self.assertNotIn("sliding_window", catalog[0])
            models = manager.list_engine_models()
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]["path"], str(engine_path))
            runtime = manager.resolve_model_runtime_settings_for_path(str(engine_path))
            self.assertEqual(runtime["inference_width"], 1024)
            self.assertEqual(runtime["inference_height"], 768)
            self.assertNotIn("sliding_window", runtime)

    def test_upload_rf_zip_package_extracts_artifacts_and_package_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(temp_dir)
            uploaded = manager.upload_model(
                "rf",
                FakeUpload("sample-rf.zip", self.make_rf_package_zip("sample-rf")),
                owner_user_id=7,
            )

            model_dir = Path(temp_dir, "rf")
            self.assertEqual(uploaded["name"], "sample-rf")
            self.assertEqual(uploaded["path"], str(model_dir / "sample-rf.onnx"))
            self.assertTrue((model_dir / "sample-rf.onnx").is_file())
            self.assertTrue((model_dir / "sample-rf.inference_config.json").is_file())
            self.assertTrue((model_dir / "sample-rf.model_config.json").is_file())
            self.assertTrue((model_dir / "sample-rf.class_names.txt").is_file())
            self.assertFalse((model_dir / "sample-rf.zip").exists())

            metadata = uploaded["package_metadata"]
            self.assertEqual(metadata["inference_width"], 960)
            self.assertEqual(metadata["inference_height"], 960)
            self.assertEqual(metadata["preprocessing"]["color_mode"], "rgb")
            self.assertEqual(metadata["preprocessing"]["resize_mode"], "stretch")
            self.assertEqual(metadata["preprocessing"]["scaling_factor"], 255)
            self.assertEqual(
                metadata["preprocessing"]["normalization"],
                [
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ],
            )

            catalog = manager.build_model_catalog()
            self.assertEqual(len(catalog), 1)
            self.assertEqual(catalog[0]["type"], "rf")
            self.assertEqual(catalog[0]["sources"], ["onnx"])
            self.assertEqual(catalog[0]["inference_width"], 960)
            self.assertEqual(catalog[0]["inference_height"], 960)
            self.assertEqual(catalog[0]["package_metadata"]["inference_width"], 960)

    def test_rf_onnx_package_is_compilable_without_changing_ul_pt_selection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(temp_dir)
            rf_source = self.touch_model_source(temp_dir, "rf/sample.onnx")
            rf_engine = self.touch_model_source(temp_dir, "rf/sample-fp16.engine")
            Path(temp_dir, "rf/sample.inference_config.json").write_text(
                json.dumps({"network_input": {"training_input_size": {"width": 960, "height": 960}}}),
                encoding="utf-8",
            )
            Path(temp_dir, "rf/sample.class_names.txt").write_text("tilletia\n", encoding="utf-8")
            ul_source = self.touch_model_source(temp_dir, "ul/sample.pt")

            catalog = manager.build_model_catalog()
            rf_entry = next(item for item in catalog if item["type"] == "rf")
            ul_entry = next(item for item in catalog if item["type"] == "ul")

            self.assertEqual(manager._find_compile_source_path(rf_entry), str(rf_source))
            self.assertEqual(manager._find_compile_source_path(ul_entry), str(ul_source))
            self.assertTrue(rf_entry["compiled"])
            self.assertEqual(rf_entry["engine"]["path"], str(rf_engine))
            self.assertEqual(len([item for item in catalog if item["type"] == "rf"]), 1)
            self.assertEqual(
                manager._build_compile_command("rf", str(rf_source)),
                [sys.executable, str(Path(temp_dir) / "convert.py"), "--model", str(rf_source)],
            )

            job_id = "rf-job"
            manager.compile_jobs[job_id] = manager._new_compile_job(
                job_id,
                "rf",
                "sample",
                str(rf_source),
            )
            manager.compile_jobs[job_id]["command"] = ["convert", "--model", str(rf_source)]
            manager._record_compile_metadata(job_id)
            self.assertEqual(
                json.loads(Path(f"{rf_engine}.json").read_text(encoding="utf-8")),
                {"network_input": {"training_input_size": {"width": 960, "height": 960}}},
            )
            self.assertEqual(
                Path(f"{rf_engine}.class_names.txt").read_text(encoding="utf-8"),
                "tilletia\n",
            )

    def test_rf_engine_runtime_settings_use_engine_sidecar_config(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(temp_dir)
            rf_source = self.touch_model_source(temp_dir, "rf/sample.onnx")
            rf_engine = self.touch_model_source(temp_dir, "rf/sample-fp16.engine")
            Path(temp_dir, "rf/sample.inference_config.json").write_text(
                json.dumps({"network_input": {"training_input_size": {"width": 800, "height": 800}}}),
                encoding="utf-8",
            )
            Path(f"{rf_engine}.json").write_text(
                json.dumps({"network_input": {"training_input_size": {"width": 800, "height": 800}}}),
                encoding="utf-8",
            )

            runtime = manager.resolve_model_runtime_settings_for_path(str(rf_engine))
            self.assertEqual(runtime["inference_width"], 800)
            self.assertEqual(runtime["inference_height"], 800)

            catalog = manager.build_model_catalog()
            rf_entry = next(item for item in catalog if item["type"] == "rf")
            self.assertEqual(manager._find_compile_source_path(rf_entry), str(rf_source))
            self.assertEqual(rf_entry["engine"]["path"], str(rf_engine))
            self.assertEqual(rf_entry["inference_width"], 800)
            self.assertEqual(rf_entry["inference_height"], 800)

    def test_tilletia_filter_defaults_are_exposed_for_unconfigured_models(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(temp_dir)
            source_path = self.touch_model_source(temp_dir, "ul/sample.pt")

            runtime = manager.resolve_model_runtime_settings_for_path(str(source_path))
            catalog = manager.build_model_catalog()

            expected = {
                "tilletia_filter_max_width_px": 68,
                "tilletia_filter_max_height_px": 68,
                "tilletia_filter_training_width": 2592,
                "tilletia_filter_training_height": 1944,
            }
            for key, value in expected.items():
                self.assertEqual(runtime[key], value)
                self.assertEqual(catalog[0][key], value)

    def test_tilletia_filter_settings_persist_per_model_and_resolve_for_engine_aliases(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(temp_dir)
            ul_source = self.touch_model_source(temp_dir, "ul/sample.pt")
            rf_source = self.touch_model_source(temp_dir, "rf/sample-rf.onnx")
            rf_engine = self.touch_model_source(temp_dir, "rf/sample-rf-fp16.engine")
            ul_values = {
                "tilletia_filter_max_width_px": 80,
                "tilletia_filter_max_height_px": 42,
                "tilletia_filter_training_width": 2048,
                "tilletia_filter_training_height": 1536,
            }
            rf_values = {
                "tilletia_filter_max_width_px": 91,
                "tilletia_filter_max_height_px": 55,
                "tilletia_filter_training_width": 3000,
                "tilletia_filter_training_height": 2000,
            }

            manager.set_model_metadata("ul", "sample", ul_values)
            manager.set_model_metadata("rf", "sample-rf", rf_values)

            ul_runtime = manager.resolve_model_runtime_settings_for_path(str(ul_source))
            rf_source_runtime = manager.resolve_model_runtime_settings_for_path(str(rf_source))
            rf_engine_runtime = manager.resolve_model_runtime_settings_for_path(str(rf_engine))
            for key, value in ul_values.items():
                self.assertEqual(ul_runtime[key], value)
            for key, value in rf_values.items():
                self.assertEqual(rf_source_runtime[key], value)
                self.assertEqual(rf_engine_runtime[key], value)

    def test_tilletia_filter_updates_require_four_positive_integers_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(temp_dir)
            self.touch_model_source(temp_dir, "ul/sample.pt")
            valid_values = {
                "tilletia_filter_max_width_px": 80,
                "tilletia_filter_max_height_px": 42,
                "tilletia_filter_training_width": 2048,
                "tilletia_filter_training_height": 1536,
            }
            manager.set_model_metadata("ul", "sample", valid_values)
            metadata_path = Path(temp_dir, "model_metadata.json")
            saved_metadata = metadata_path.read_text(encoding="utf-8")

            invalid_payloads = [
                {"tilletia_filter_max_width_px": 90},
                {**valid_values, "tilletia_filter_max_width_px": 0},
                {**valid_values, "tilletia_filter_max_height_px": -1},
                {**valid_values, "tilletia_filter_training_width": 1.5},
                {**valid_values, "tilletia_filter_training_height": "bad"},
            ]
            for payload in invalid_payloads:
                with self.assertRaises(ModelValidationError):
                    manager.set_model_metadata("ul", "sample", payload)
                self.assertEqual(
                    metadata_path.read_text(encoding="utf-8"),
                    saved_metadata,
                )


if __name__ == "__main__":
    unittest.main()
