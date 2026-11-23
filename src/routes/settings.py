import flask

import auth
import dashboard_settings
import mailer
import roboflow_settings
import secret_store
import smtp_settings


bp = flask.Blueprint("settings", __name__)


@bp.route("/api/config")
@auth.require_permission("dashboard:view")
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
    return flask.jsonify({"stream_port": 8889})


@bp.route("/api/dashboard-settings", methods=["GET"])
@auth.require_permission("dashboard_settings:view")
def api_get_dashboard_settings():
    """
    Get the persisted dashboard settings used as defaults for new runs.
    ---
    tags:
      - Configuration
    responses:
      200:
        description: Current dashboard settings.
        schema:
          type: object
          properties:
            site_name:
              type: string
              description: Enterprise (elevator) name shown in the page header.
            analysis_number:
              type: string
            source_type:
              type: string
              enum: ['camera', 'file']
            camera_device:
              type: string
            camera_mode:
              type: object
              properties:
                width:
                  type: integer
                height:
                  type: integer
                fps:
                  type: integer
                format:
                  type: string
            uploaded_path:
              type: string
            model_path:
              type: string
            vis_conf:
              type: number
            captions_enabled:
              type: boolean
              description: Render class, confidence, and tracker-ID captions on detection boxes. Defaults to false.
            grid_count_enabled:
              type: boolean
            grid_debug_enabled:
              type: boolean
            grid_score_threshold:
              type: number
            ask_manual_spore_count:
              type: boolean
            max_run_minutes:
              type: integer
              description: Run duration limit in minutes after which a run is stopped automatically. Defaults to 30.
      401:
        description: Missing or invalid access token.
      403:
        description: Authenticated user lacks permission to view dashboard settings.
    """
    return flask.jsonify(dashboard_settings.get_dashboard_settings())


@bp.route("/api/dashboard-settings", methods=["PUT"])
@auth.require_permission("dashboard:configure")
def api_put_dashboard_settings():
    """
    Update the persisted dashboard settings used as defaults for new runs.
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
          properties:
            site_name:
              type: string
              description: Enterprise (elevator) name shown in the page header.
            analysis_number:
              type: string
            source_type:
              type: string
              enum: ['camera', 'file']
            camera_device:
              type: string
            camera_mode:
              type: object
              properties:
                width:
                  type: integer
                height:
                  type: integer
                fps:
                  type: integer
                format:
                  type: string
            uploaded_path:
              type: string
            model_path:
              type: string
            vis_conf:
              type: number
            captions_enabled:
              type: boolean
              description: Render class, confidence, and tracker-ID captions on detection boxes. Defaults to false.
            grid_count_enabled:
              type: boolean
            grid_debug_enabled:
              type: boolean
            grid_score_threshold:
              type: number
            ask_manual_spore_count:
              type: boolean
            max_run_minutes:
              type: integer
              description: Run duration limit in minutes (1-480) after which a run is stopped automatically. Invalid or empty values fall back to 30.
    responses:
      200:
        description: Updated dashboard settings.
      400:
        description: Invalid settings payload.
      401:
        description: Missing or invalid access token.
      403:
        description: Authenticated user lacks permission to update dashboard settings.
    """
    payload = flask.request.get_json(silent=True) or {}
    try:
        settings = dashboard_settings.update_dashboard_settings(payload)
    except (TypeError, ValueError) as exc:
        return flask.jsonify({"error": str(exc)}), 400
    return flask.jsonify(settings)


@bp.route("/api/roboflow/settings", methods=["GET"])
@auth.require_permission("roboflow:manage")
def api_get_roboflow_settings():
    """
    Get the server-side Roboflow upload configuration without exposing the API token.
    ---
    tags:
      - Roboflow
    responses:
      200:
        description: Safe Roboflow configuration status.
        schema:
          type: object
          properties:
            project_name:
              type: string
            api_token_configured:
              type: boolean
            api_token_usable:
              type: boolean
            secure_storage_status:
              type: string
              enum: [available, unavailable, error]
            secure_storage_available:
              type: boolean
            api_token_storage:
              type: string
              enum: [none, memory, encrypted, encrypted_unavailable, plaintext_file, plaintext_file_unconfirmed, plaintext_file_unavailable]
            api_token_persistent:
              type: boolean
      401:
        description: Missing or invalid access token.
      403:
        description: Authenticated user is not an administrator.
    """
    return flask.jsonify(roboflow_settings.get_roboflow_settings())


@bp.route("/api/roboflow/settings", methods=["PUT"])
@auth.require_permission("roboflow:manage")
def api_put_roboflow_settings():
    """
    Save the server-side Roboflow upload configuration.
    ---
    tags:
      - Roboflow
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            project_name:
              type: string
              description: Roboflow project ID, also called the dataset name by the upload API.
            api_token:
              type: string
              description: Replacement private API token. A blank or omitted value preserves the current token.
            allow_insecure_file_storage:
              type: boolean
              description: When secure storage is unavailable, true selects an owner-only plaintext token file and false selects process memory. If omitted, an existing mode is preserved; a new token defaults to memory.
    responses:
      200:
        description: Safe Roboflow configuration status.
      400:
        description: Invalid or incomplete configuration.
      401:
        description: Missing or invalid access token.
      403:
        description: Authenticated user is not an administrator.
      503:
        description: The configured token storage cannot be used without a replacement token.
    """
    payload = flask.request.get_json(silent=True)
    try:
        settings = roboflow_settings.update_roboflow_settings(payload)
    except secret_store.SecretStorageError as exc:
        return flask.jsonify({"error": str(exc)}), 503
    except (TypeError, ValueError) as exc:
        return flask.jsonify({"error": str(exc)}), 400
    return flask.jsonify(settings)


@bp.route("/api/smtp/settings", methods=["GET"])
@auth.require_permission("smtp:manage")
def api_get_smtp_settings():
    """
    Get the server-side SMTP configuration without exposing the password.
    ---
    tags:
      - Email
    responses:
      200:
        description: Safe SMTP configuration status.
        schema:
          type: object
          properties:
            host:
              type: string
            port:
              type: integer
            username:
              type: string
            sender:
              type: string
            tls_mode:
              type: string
              enum: [starttls, ssl, none]
            password_configured:
              type: boolean
            password_storage:
              type: string
              enum: [none, env, memory, encrypted, encrypted_unavailable, plaintext_file, plaintext_file_unconfirmed, plaintext_file_unavailable]
            password_persistent:
              type: boolean
            secure_storage_available:
              type: boolean
            source:
              type: string
              enum: [none, env, db]
      403:
        description: Authenticated user cannot manage SMTP settings.
    """
    return flask.jsonify(smtp_settings.get_smtp_settings())


@bp.route("/api/smtp/settings", methods=["PUT"])
@auth.require_permission("smtp:manage")
def api_put_smtp_settings():
    """
    Save the server-side SMTP configuration.
    ---
    tags:
      - Email
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            host:
              type: string
            port:
              type: integer
            username:
              type: string
            sender:
              type: string
              description: From address used for outgoing mail.
            tls_mode:
              type: string
              enum: [starttls, ssl, none]
            password:
              type: string
              description: Replacement password. A blank or omitted value preserves the current password.
            allow_insecure_file_storage:
              type: boolean
              description: Allow storing the password in an owner-only plaintext file when no secure storage is available.
    responses:
      200:
        description: Safe SMTP configuration status.
      400:
        description: Invalid or incomplete configuration.
      403:
        description: Authenticated user cannot manage SMTP settings.
    """
    payload = flask.request.get_json(silent=True)
    try:
        settings = smtp_settings.update_smtp_settings(payload)
    except secret_store.SecretStorageError as exc:
        return flask.jsonify({"error": str(exc)}), 503
    except (TypeError, ValueError) as exc:
        return flask.jsonify({"error": str(exc)}), 400
    return flask.jsonify(settings)


@bp.route("/api/smtp/test", methods=["POST"])
@auth.require_permission("smtp:manage")
def api_test_smtp_settings():
    """
    Send a short test email using the supplied settings, falling back to the saved ones.
    ---
    tags:
      - Email
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: false
        schema:
          type: object
          properties:
            to:
              type: string
              description: Optional recipient. Defaults to the effective sender address.
            host:
              type: string
              description: Overrides the saved SMTP host for this test only.
            port:
              type: integer
              description: Overrides the saved port for this test only.
            username:
              type: string
              description: Overrides the saved login for this test only.
            sender:
              type: string
              description: Overrides the saved From address for this test only.
            tls_mode:
              type: string
              enum: [starttls, ssl, none]
              description: Overrides the saved encryption mode for this test only.
            password:
              type: string
              description: Password used for this test. A blank value reuses the saved password.
    responses:
      200:
        description: Test email sent successfully.
      400:
        description: Invalid recipient or incomplete SMTP configuration.
      403:
        description: Authenticated user cannot manage SMTP settings.
      503:
        description: The stored password is unavailable or the test send failed.
    """
    payload = flask.request.get_json(silent=True) or {}

    try:
        settings = smtp_settings.resolve_smtp_settings(payload)
    except secret_store.SecretStorageError as exc:
        return flask.jsonify({"error": str(exc)}), 503
    except (TypeError, ValueError) as exc:
        return flask.jsonify({"error": str(exc)}), 400

    if not settings["host"]:
        return flask.jsonify({"error": "Вкажіть SMTP-сервер."}), 400
    if not mailer.EMAIL_ADDRESS_RE.match(settings["sender"]):
        return flask.jsonify({"error": "Вкажіть коректну адресу відправника."}), 400

    recipient = str(payload.get("to") or settings["sender"]).strip()
    if not mailer.EMAIL_ADDRESS_RE.match(recipient):
        return flask.jsonify({"error": "Вкажіть коректну email-адресу для перевірки."}), 400

    subject = "Tilletia: перевірка SMTP"
    body = (
        "Це тестовий лист від Tilletia.\n"
        "Якщо ви його отримали, налаштування пошти працюють."
    )
    try:
        mailer.send_email(settings, [recipient], subject, body, attachments=[])
    except RuntimeError as exc:
        return flask.jsonify({"error": str(exc)}), 503

    return flask.jsonify({"success": True, "to": recipient})
