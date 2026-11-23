import flask

import auth
import page_context
import roboflow_settings
import smtp_settings


bp = flask.Blueprint("pages", __name__)


@bp.route("/settings")
@auth.require_permission(html_redirect=True)
def settings_page():
    """
    Serve the authenticated user settings page.
    ---
    tags:
      - Pages
    produces:
      - text/html
    responses:
      200:
        description: User settings HTML.
      302:
        description: Redirect to the login page when the caller is not authenticated.
      401:
        description: Missing or invalid access token.
    """
    return flask.render_template(
        "settings.html",
        role_label=", ".join(flask.g.current_user.get("roles", [])),
        roboflow_settings=(
            roboflow_settings.get_roboflow_settings()
            if auth.user_has_permission(flask.g.current_user, "roboflow:manage")
            else None
        ),
        smtp_settings=(
            smtp_settings.get_smtp_settings()
            if auth.user_has_permission(flask.g.current_user, "smtp:manage")
            else None
        ),
        **page_context.build_page_context(flask.g.current_user),
    )


@bp.route("/")
@auth.require_permission("dashboard:view", html_redirect=True)
def index():
    """
    Serve the main dashboard page.
    ---
    tags:
      - Pages
    produces:
      - text/html
    responses:
      200:
        description: Dashboard HTML.
      302:
        description: Redirect to the login page when the caller is not authenticated.
      403:
        description: Authenticated user lacks permission to view the dashboard.
    """
    return flask.render_template("index.html", **page_context.build_page_context(flask.g.current_user))


@bp.route("/results")
@auth.require_permission("results:view", html_redirect=True)
def results_page():
    """
    Serve the results page.
    ---
    tags:
      - Pages
    produces:
      - text/html
    responses:
      200:
        description: Results page HTML.
      302:
        description: Redirect to the login page when the caller is not authenticated.
      403:
        description: Authenticated user lacks permission to view results.
    """
    return flask.render_template("results.html", **page_context.build_page_context(flask.g.current_user))


@bp.route("/models")
@auth.require_permission("models:view", html_redirect=True)
def models_page():
    """
    Serve the model catalog page.
    ---
    tags:
      - Pages
    produces:
      - text/html
    responses:
      200:
        description: Model catalog HTML.
      302:
        description: Redirect to the login page when the caller is not authenticated.
      403:
        description: Authenticated user lacks permission to view the model catalog.
    """
    return flask.render_template("models.html", **page_context.build_page_context(flask.g.current_user))
