import os

import flask

import auth
import stage
import stage_link


bp = flask.Blueprint("stage", __name__)
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _call(action):
    try:
        return flask.jsonify(action())
    except stage.StageError as exc:
        return flask.jsonify({"error": str(exc)}), 409
    except stage_link.ModbusError as exc:
        return flask.jsonify({"error": str(exc)}), 503
    except (KeyError, TypeError, ValueError) as exc:
        message = str(exc) if str(exc) else "Невірні дані"
        return flask.jsonify({"error": message}), 400


@bp.route("/stage-panel.js")
def stage_panel_asset():
    return flask.send_from_directory(SRC_DIR, "stage_panel.js", mimetype="application/javascript")


@bp.route("/api/stage", methods=["GET"])
@auth.require_permission("stage:view")
def api_stage_status():
    return _call(lambda: stage.get_stage().status())


@bp.route("/api/stage/settings", methods=["PUT"])
@auth.require_permission("stage:move")
def api_stage_settings():
    body = flask.request.get_json(silent=True) or {}
    return _call(lambda: stage.get_stage().update_settings(
        speed_mm_s=body.get("speed_mm_s"),
        step_mm=body.get("step_mm"),
        workspace_mm=body.get("workspace_mm"),
    ))


@bp.route("/api/stage/jog", methods=["POST"])
@auth.require_permission("stage:move")
def api_stage_jog():
    body = flask.request.get_json(silent=True) or {}
    return _call(lambda: stage.get_stage().jog(body.get("axis"), body.get("sign", 1)))


@bp.route("/api/stage/stop", methods=["POST"])
@auth.require_permission("stage:move")
def api_stage_stop():
    return _call(lambda: stage.get_stage().stop())


@bp.route("/api/stage/zero", methods=["POST"])
@auth.require_permission("stage:move")
def api_stage_zero():
    return _call(lambda: stage.get_stage().set_zero())


@bp.route("/api/stage/points", methods=["POST"])
@auth.require_permission("stage:move")
def api_stage_add_point():
    body = flask.request.get_json(silent=True) or {}
    point = body.get("point")
    return _call(lambda: stage.get_stage().add_point(point))


@bp.route("/api/stage/points/last", methods=["DELETE"])
@auth.require_permission("stage:move")
def api_stage_pop_point():
    return _call(lambda: stage.get_stage().pop_point())


@bp.route("/api/stage/points", methods=["PUT"])
@auth.require_permission("stage:move")
def api_stage_replace_points():
    body = flask.request.get_json(silent=True) or {}
    return _call(lambda: stage.get_stage().replace_points(body.get("points")))


@bp.route("/api/stage/goryaev", methods=["POST"])
@auth.require_permission("stage:move")
def api_stage_goryaev():
    body = flask.request.get_json(silent=True) or {}
    return _call(lambda: stage.get_stage().fill_goryaev(
        step_mm=body.get("step_mm", 0.5),
        width_mm=body.get("width_mm", 3.0),
        height_mm=body.get("height_mm", 3.0),
        second_offset=body.get("second_offset"),
    ))


@bp.route("/api/stage/routes", methods=["POST"])
@auth.require_permission("stage:move")
def api_stage_save_route():
    body = flask.request.get_json(silent=True) or {}
    return _call(lambda: stage.get_stage().save_route(body.get("name")))


@bp.route("/api/stage/routes/<int:route_id>/load", methods=["POST"])
@auth.require_permission("stage:move")
def api_stage_load_route(route_id):
    return _call(lambda: stage.get_stage().load_route(route_id))


@bp.route("/api/stage/routes/<int:route_id>", methods=["DELETE"])
@auth.require_permission("stage:move")
def api_stage_delete_route(route_id):
    return _call(lambda: stage.get_stage().delete_route(route_id))


@bp.route("/api/stage/run", methods=["POST"])
@auth.require_permission("stage:move")
def api_stage_run():
    body = flask.request.get_json(silent=True) or {}
    return _call(lambda: stage.get_stage().run_points(speed_mm_s=body.get("speed_mm_s")))
