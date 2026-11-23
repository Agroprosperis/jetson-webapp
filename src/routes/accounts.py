import os

import flask

import auth
import cookies
import page_context
import tokens


SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

bp = flask.Blueprint("accounts", __name__)


@bp.route("/login")
def login_page():
    """
    Serve the login page.
    ---
    tags:
      - Pages
    produces:
      - text/html
    responses:
      200:
        description: Login page HTML.
      302:
        description: Redirect to the dashboard when the caller is already authenticated.
    """
    user = auth.load_request_user()
    if user is not None:
        return flask.redirect(flask.url_for("pages.index"))
    return flask.render_template("login.html")


@bp.route("/auth-client.js")
def auth_client_asset():
    return flask.send_from_directory(SRC_DIR, "auth_client.js", mimetype="application/javascript")


@bp.route("/auth/login", methods=["POST"])
def auth_login():
    """
    Authenticate a local user and issue bearer tokens.
    ---
    tags:
      - Auth
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - username
            - password
          properties:
            username:
              type: string
            password:
              type: string
    responses:
      200:
        description: Access token, refresh token, and authenticated user identity.
        schema:
          type: object
          properties:
            access_token:
              type: string
            refresh_token:
              type: string
            user:
              type: object
              properties:
                id:
                  type: integer
                username:
                  type: string
      401:
        description: Invalid username or password.
      403:
        description: Password change is required before first login.
    """
    payload = flask.request.get_json(silent=True) or {}
    user = auth.authenticate_user(payload.get("username", ""), payload.get("password", ""))
    if user is None:
        return flask.jsonify({"error": "Invalid username or password"}), 401
    if user.get("force_password_change"):
        return (
            flask.jsonify(
                {
                    "error": "Password change required.",
                    "password_change_required": True,
                }
            ),
            403,
        )

    access_token = tokens.issue_access_token(user["id"])
    refresh_token = auth.issue_refresh_token(user["id"])
    response = flask.jsonify(
        {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user": {
                "id": user["id"],
                "username": user["username"],
            },
        }
    )
    return cookies.set_auth_cookies(
        response,
        access_token,
        refresh_token,
        access_max_age=tokens.ACCESS_TOKEN_TTL_SECONDS,
        refresh_max_age=tokens.REFRESH_TOKEN_TTL_SECONDS,
    )


@bp.route("/auth/refresh", methods=["POST"])
def auth_refresh():
    """
    Exchange a refresh token for a new bearer access token.
    ---
    tags:
      - Auth
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: false
        schema:
          type: object
          properties:
            refresh_token:
              type: string
              description: Optional refresh token. When omitted, the refresh cookie is used.
    responses:
      200:
        description: Refreshed access token, rotated refresh token, and authenticated user identity.
        schema:
          type: object
          properties:
            access_token:
              type: string
            refresh_token:
              type: string
            user:
              type: object
              properties:
                id:
                  type: integer
                username:
                  type: string
      401:
        description: Missing or invalid refresh token.
    """
    payload = flask.request.get_json(silent=True) or {}
    refresh_token = (payload.get("refresh_token") or cookies.get_refresh_token_from_request(flask.request)).strip()
    if not refresh_token:
        return flask.jsonify({"error": "Missing refresh token"}), 401

    rotated = auth.rotate_refresh_token(refresh_token)
    if rotated is None:
        return flask.jsonify({"error": "Invalid refresh token"}), 401

    user = rotated["user"]
    access_token = tokens.issue_access_token(user["id"])
    response = flask.jsonify(
        {
            "access_token": access_token,
            "refresh_token": rotated["refresh_token"],
            "user": {
                "id": user["id"],
                "username": user["username"],
            },
        }
    )
    return cookies.set_auth_cookies(
        response,
        access_token,
        rotated["refresh_token"],
        access_max_age=tokens.ACCESS_TOKEN_TTL_SECONDS,
        refresh_max_age=tokens.REFRESH_TOKEN_TTL_SECONDS,
    )


@bp.route("/auth/change-password", methods=["POST"])
def auth_change_password():
    """
    Change a local user's password and clear the first-login password-change requirement.
    ---
    tags:
      - Auth
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - username
            - current_password
            - new_password
            - confirm_password
          properties:
            username:
              type: string
            current_password:
              type: string
            new_password:
              type: string
            confirm_password:
              type: string
    responses:
      200:
        description: Password updated successfully.
        schema:
          type: object
          properties:
            success:
              type: boolean
      400:
        description: Invalid password-change payload.
      401:
        description: Invalid username or current password.
    """
    payload = flask.request.get_json(silent=True) or {}
    if payload.get("new_password") != payload.get("confirm_password"):
        return flask.jsonify({"error": "Passwords do not match"}), 400
    try:
        user = auth.change_password(
            payload.get("username"),
            payload.get("current_password"),
            payload.get("new_password"),
        )
    except ValueError as exc:
        return flask.jsonify({"error": str(exc)}), 400
    if user is None:
        return flask.jsonify({"error": "Invalid username or current password"}), 401
    return flask.jsonify({"success": True})


@bp.route("/auth/logout", methods=["POST"])
@auth.require_permission()
def auth_logout():
    """
    Revoke the current refresh token and clear auth cookies.
    ---
    tags:
      - Auth
    responses:
      200:
        description: Logout completed successfully.
        schema:
          type: object
          properties:
            success:
              type: boolean
      401:
        description: Missing or invalid access token.
    """
    flask.g._skip_auth_cookie_refresh = True
    auth.revoke_refresh_token(cookies.get_refresh_token_from_request(flask.request))
    response = flask.jsonify({"success": True})
    return cookies.clear_auth_cookies(response)


@bp.route("/auth/me")
@auth.require_permission()
def auth_me():
    """
    Return the authenticated user identity.
    ---
    tags:
      - Auth
    responses:
      200:
        description: Authenticated user identity.
        schema:
          type: object
          properties:
            id:
              type: integer
            username:
              type: string
      401:
        description: Missing or invalid access token.
    """
    user = flask.g.current_user
    return flask.jsonify(
        {
            "id": user["id"],
            "username": user["username"],
        }
    )


@bp.route("/users", methods=["POST"])
@auth.require_permission("users:manage")
def create_user_route():
    """
    Create a local user with one or more roles.
    ---
    tags:
      - Auth
    consumes:
      - application/json
    parameters:
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - username
            - password
            - roles
          properties:
            username:
              type: string
            password:
              type: string
            roles:
              type: array
              items:
                type: string
                enum: ['admin', 'engineer', 'user', 'viewer']
    responses:
      201:
        description: Created user summary.
        schema:
          type: object
          properties:
            id:
              type: integer
            username:
              type: string
      400:
        description: Invalid payload.
      401:
        description: Missing or invalid access token.
      403:
        description: Authenticated user lacks permission to manage users.
      409:
        description: Username already exists.
    """
    payload = flask.request.get_json(silent=True) or {}
    try:
        user = auth.create_user(
            payload.get("username"),
            payload.get("password"),
            payload.get("roles"),
        )
    except ValueError as exc:
        message = str(exc)
        status = 409 if "exists" in message.lower() else 400
        return flask.jsonify({"error": message}), status

    return (
        flask.jsonify(
            {
                "id": user["id"],
                "username": user["username"],
            }
        ),
        201,
    )


@bp.route("/users/<int:user_id>", methods=["PATCH"])
@auth.require_permission("users:manage")
def update_user_route(user_id):
    """
    Change a user's roles and/or active status.
    ---
    tags:
      - Auth
    consumes:
      - application/json
    parameters:
      - name: user_id
        in: path
        type: integer
        required: true
      - name: body
        in: body
        required: true
        schema:
          type: object
          properties:
            roles:
              type: array
              items:
                type: string
                enum: ['admin', 'engineer', 'user', 'viewer']
            active:
              type: boolean
              description: Deactivated users cannot log in and their sessions are revoked.
    responses:
      200:
        description: Updated user summary.
      400:
        description: Invalid payload, or an admin tried to deactivate or demote themselves.
      401:
        description: Missing or invalid access token.
      403:
        description: Authenticated user lacks permission to manage users.
      404:
        description: User not found.
    """
    payload = flask.request.get_json(silent=True) or {}
    try:
        user = auth.update_user(
            user_id,
            roles=payload.get("roles"),
            active=payload.get("active"),
            acting_user_id=flask.g.current_user["id"],
        )
    except ValueError as exc:
        return flask.jsonify({"error": str(exc)}), 400
    if user is None:
        return flask.jsonify({"error": "User not found."}), 404
    return flask.jsonify(
        {
            "id": user["id"],
            "username": user["username"],
            "roles": user["roles"],
            "active": user["active"],
        }
    )


@bp.route("/users/<int:user_id>/reset-password", methods=["POST"])
@auth.require_permission("users:manage")
def reset_user_password_route(user_id):
    """
    Set a temporary password that the user must change on next login.
    ---
    tags:
      - Auth
    consumes:
      - application/json
    parameters:
      - name: user_id
        in: path
        type: integer
        required: true
      - name: body
        in: body
        required: true
        schema:
          type: object
          required:
            - password
          properties:
            password:
              type: string
    responses:
      200:
        description: Password reset; existing sessions are revoked.
      400:
        description: Missing password.
      401:
        description: Missing or invalid access token.
      403:
        description: Authenticated user lacks permission to manage users.
      404:
        description: User not found.
    """
    payload = flask.request.get_json(silent=True) or {}
    try:
        user = auth.reset_user_password(user_id, payload.get("password"))
    except ValueError as exc:
        return flask.jsonify({"error": str(exc)}), 400
    if user is None:
        return flask.jsonify({"error": "User not found."}), 404
    return flask.jsonify({"success": True})


@bp.route("/users")
@auth.require_permission("users:manage", html_redirect=True)
def users_page():
    """
    Serve the user-management page.
    ---
    tags:
      - Pages
    produces:
      - text/html
    responses:
      200:
        description: User-management HTML.
      302:
        description: Redirect to the login page when the caller is not authenticated.
      403:
        description: Authenticated user lacks permission to manage users.
    """
    return flask.render_template(
        "users.html",
        **page_context.build_page_context(flask.g.current_user, users=auth.list_users()),
    )
