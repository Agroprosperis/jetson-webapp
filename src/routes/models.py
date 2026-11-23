import logging

import flask

import auth
import ownership

from model_manager import (
    ModelConflictError,
    ModelManager,
    ModelManagerError,
    ModelNotFoundError,
    ModelValidationError,
)


LOGGER = logging.getLogger("app")

model_manager = ModelManager(auth_module=ownership, logger=LOGGER)

bp = flask.Blueprint("models", __name__)


@bp.route("/api/models")
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


@bp.route("/api/model-catalog")
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
                  tilletia_filter_max_width_px:
                    type: integer
                    description: Maximum allowed Tilletia width in training-image pixels.
                  tilletia_filter_max_height_px:
                    type: integer
                    description: Maximum allowed Tilletia height in training-image pixels.
                  tilletia_filter_training_width:
                    type: integer
                    description: Training-image width used to scale the filter.
                  tilletia_filter_training_height:
                    type: integer
                    description: Training-image height used to scale the filter.
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


@bp.route("/api/model-upload", methods=["POST"])
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


@bp.route("/api/model-compile", methods=["POST"])
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


@bp.route("/api/model-task", methods=["POST"])
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


@bp.route("/api/model-metadata", methods=["POST"])
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
            tilletia_filter_max_width_px:
              type: integer
              minimum: 1
              description: Maximum allowed Tilletia width in training-image pixels.
            tilletia_filter_max_height_px:
              type: integer
              minimum: 1
              description: Maximum allowed Tilletia height in training-image pixels.
            tilletia_filter_training_width:
              type: integer
              minimum: 1
              description: Width of the original training images.
            tilletia_filter_training_height:
              type: integer
              minimum: 1
              description: Height of the original training images.
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


@bp.route("/api/model-compile-jobs")
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


@bp.route("/api/model-compile/<job_id>")
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
