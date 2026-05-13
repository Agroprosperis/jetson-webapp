#!/usr/bin/env python3
import argparse
import json
import os
import re
import shutil
import sys
import zipfile
from pathlib import Path


DEFAULT_MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/tmp/cache")
DEFAULT_OUTPUT_DIR = os.getenv("DEPLOY_MODEL_OUTPUT_DIR", "data/model/rf")
WEIGHTS_FILENAME = "weights.onnx"
INFERENCE_CONFIG_FILENAME = "inference_config.json"
MODEL_CONFIG_FILENAME = "model_config.json"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download a Roboflow ONNX model into a fixed cache and stage deployment artifacts."
    )
    parser.add_argument("model_id", help="Roboflow model id, for example project-name/1")
    parser.add_argument("roboflow_secret", help="Roboflow API key/secret used to download the model")
    parser.add_argument(
        "--cache-dir",
        default=DEFAULT_MODEL_CACHE_DIR,
        help=f"Roboflow inference cache root. Default: {DEFAULT_MODEL_CACHE_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help=f"Local model output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--image-width",
        type=int,
        default=960,
        help="Width of the generated blank smoke-test image.",
    )
    parser.add_argument(
        "--image-height",
        type=int,
        default=960,
        help="Height of the generated blank smoke-test image.",
    )
    parser.add_argument(
        "--no-zip",
        action="store_true",
        help="Stage files only; do not create a zip archive.",
    )
    return parser.parse_args()


def safe_model_name(model_id):
    name = model_id.strip().replace("/", "-")
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name)
    name = name.strip(".-_")
    if not name:
        raise ValueError("model_id does not produce a safe filename")
    return name


def configure_roboflow_environment(cache_dir, roboflow_secret):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    os.environ["MODEL_CACHE_DIR"] = str(cache_dir)
    os.environ.setdefault("TENSORRT_CACHE_PATH", str(cache_dir))
    os.environ.setdefault("ONNXRUNTIME_EXECUTION_PROVIDERS", "[CPUExecutionProvider]")
    os.environ["ROBOFLOW_API_KEY"] = roboflow_secret


def run_blank_image_inference(model_id, roboflow_secret, image_width, image_height):
    from inference import get_model
    from PIL import Image

    loaded_model_dir = {}

    def point_model_directory(model_dir):
        loaded_model_dir["path"] = model_dir

    model = get_model(
        model_id=model_id,
        api_key=roboflow_secret,
        point_model_directory=point_model_directory,
    )
    blank_image = Image.new("RGB", (image_width, image_height), color=(0, 0, 0))
    model.infer(blank_image)
    return model, loaded_model_dir.get("path")


def require_file(path, description):
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def copy_artifacts(cache_model_dir, output_dir, package_name):
    output_dir.mkdir(parents=True, exist_ok=True)

    source_weights = require_file(
        cache_model_dir / WEIGHTS_FILENAME,
        "Roboflow ONNX weights",
    )
    source_inference_config = require_file(
        cache_model_dir / INFERENCE_CONFIG_FILENAME,
        "Roboflow inference config",
    )
    source_model_config = cache_model_dir / MODEL_CONFIG_FILENAME

    staged_weights = output_dir / f"{package_name}.onnx"
    staged_inference_config = output_dir / f"{package_name}.inference_config.json"

    shutil.copy2(source_weights, staged_weights)
    shutil.copy2(source_inference_config, staged_inference_config)

    staged_files = [staged_weights, staged_inference_config]
    if source_model_config.is_file():
        staged_model_config = output_dir / f"{package_name}.model_config.json"
        shutil.copy2(source_model_config, staged_model_config)
        staged_files.append(staged_model_config)

    source_classes = cache_model_dir / "class_names.txt"
    if source_classes.is_file():
        staged_classes = output_dir / f"{package_name}.class_names.txt"
        shutil.copy2(source_classes, staged_classes)
        staged_files.append(staged_classes)

    return staged_files


def write_manifest(
    *,
    output_dir,
    package_name,
    model_id,
    resolved_model_id,
    cache_dir,
    cache_model_dir,
    staged_files,
    zip_path=None,
):
    manifest_path = output_dir / f"{package_name}.manifest.json"
    payload = {
        "model_id": model_id,
        "resolved_model_id": resolved_model_id,
        "model_cache_dir": str(cache_dir),
        "cache_model_dir": str(cache_model_dir),
        "artifacts": [path.name for path in staged_files],
        "zip": zip_path.name if zip_path else None,
    }
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def create_zip(output_dir, package_name, package_files):
    zip_path = output_dir / f"{package_name}.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in package_files:
            archive.write(path, arcname=path.name)
    return zip_path


def main():
    args = parse_args()

    try:
        package_name = safe_model_name(args.model_id)
        cache_dir = Path(args.cache_dir)
        output_dir = Path(args.output_dir)

        configure_roboflow_environment(cache_dir, args.roboflow_secret)
        _, loaded_model_dir = run_blank_image_inference(
            args.model_id,
            args.roboflow_secret,
            args.image_width,
            args.image_height,
        )

        resolved_model_id = args.model_id
        if not loaded_model_dir:
            raise RuntimeError(
                "Roboflow inference did not report a loaded model package directory."
            )
        cache_model_dir = Path(loaded_model_dir)
        staged_files = copy_artifacts(cache_model_dir, output_dir, package_name)

        zip_path = None if args.no_zip else output_dir / f"{package_name}.zip"

        manifest_path = write_manifest(
            output_dir=output_dir,
            package_name=package_name,
            model_id=args.model_id,
            resolved_model_id=resolved_model_id,
            cache_dir=cache_dir,
            cache_model_dir=cache_model_dir,
            staged_files=staged_files,
            zip_path=zip_path,
        )

        if zip_path is not None:
            zip_path = create_zip(output_dir, package_name, [*staged_files, manifest_path])

        print(f"Model cache: {cache_model_dir}")
        print("Staged artifacts:")
        for path in staged_files:
            print(f"  {path}")
        print(f"Manifest: {manifest_path}")
        if zip_path is not None:
            print(f"Zip archive: {zip_path}")

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
