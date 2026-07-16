from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from .manager import (
    ConflictError,
    ManagerError,
    ModelManager,
    NotFoundError,
    ValidationError,
)


FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"


def _strict_json(expected_fields):
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict) or set(payload) != set(expected_fields):
        raise ValidationError("Request body does not match the API contract.")
    for field in expected_fields:
        if not isinstance(payload[field], str) or not payload[field]:
            raise ValidationError(f"{field} must be a non-empty string.")
    return payload


def create_app(manager=None):
    app = Flask(__name__, static_folder=None)
    app.config["MODEL_MANAGER"] = manager or ModelManager()

    def model_manager():
        return app.config["MODEL_MANAGER"]

    @app.errorhandler(ValidationError)
    def handle_validation(exc):
        return jsonify({"error": str(exc)}), 400

    @app.errorhandler(NotFoundError)
    def handle_not_found(exc):
        return jsonify({"error": str(exc)}), 404

    @app.errorhandler(ConflictError)
    def handle_conflict(exc):
        return jsonify({"error": str(exc)}), 409

    @app.errorhandler(ManagerError)
    def handle_manager_error(exc):
        return jsonify({"error": str(exc)}), 502

    @app.get("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.get("/<path:asset_path>")
    def frontend_asset(asset_path):
        return send_from_directory(FRONTEND_DIR, asset_path)

    @app.post("/api/roboflow/catalog")
    def roboflow_catalog():
        payload = _strict_json(("api_key",))
        return jsonify(model_manager().catalog_from_roboflow(payload["api_key"]))

    @app.post("/api/roboflow/models")
    def roboflow_models():
        payload = _strict_json(("api_key", "project_id"))
        return jsonify(
            model_manager().models_from_roboflow(payload["api_key"], payload["project_id"])
        )

    @app.post("/api/models/import")
    def import_model():
        payload = _strict_json(("api_key", "model_id"))
        return jsonify(model_manager().start_import(payload["model_id"], payload["api_key"])), 202

    @app.get("/api/models")
    def models():
        return jsonify(model_manager().list_models())

    @app.post("/api/models/<model_name>/compile")
    def compile_model(model_name):
        if request.content_length not in (None, 0):
            raise ValidationError("Compile requests do not accept a body.")
        return jsonify(model_manager().start_compile(model_name)), 202

    @app.get("/api/jobs/<job_id>")
    def job(job_id):
        return jsonify(model_manager().get_job(job_id))

    @app.get("/api/models/<model_name>/package")
    def package(model_name):
        path = model_manager().create_package(model_name)
        return send_file(path, as_attachment=True, download_name=path.name)

    @app.delete("/api/models/<model_name>")
    def delete_model(model_name):
        return jsonify(model_manager().delete_model(model_name))

    return app


if __name__ == "__main__":
    create_app().run(host="0.0.0.0", port=8080)
