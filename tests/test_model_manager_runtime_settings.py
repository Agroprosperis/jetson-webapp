import json
import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_manager import ModelManager


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


if __name__ == "__main__":
    unittest.main()
