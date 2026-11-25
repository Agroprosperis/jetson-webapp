import json

import auth
import dashboard_settings
import roboflow_settings
import smtp_settings


def build_page_context(user, **extra_context):
    settings = dashboard_settings.get_dashboard_settings()
    context = {
        "current_user": user,
        "dashboard_settings": settings,
        "dashboard_settings_json": json.dumps(settings),
        "can_configure_dashboard": auth.user_has_permission(user, "dashboard:configure"),
        "can_run_pipeline": auth.user_has_permission(user, "pipeline:start"),
        "can_move_stage": auth.user_has_permission(user, "stage:move"),
        "can_annotate_results": auth.user_has_permission(user, "results:annotate"),
        "can_delete_results": auth.user_has_permission(user, "results:delete"),
        "can_view_models": auth.user_has_permission(user, "models:view"),
        "can_manage_users": auth.user_has_permission(user, "users:manage"),
        "can_manage_roboflow": auth.user_has_permission(user, "roboflow:manage"),
        "can_upload_roboflow": (
            roboflow_settings.roboflow_configured() and auth.user_has_permission(user, "roboflow:manage")
        ),
        "can_manage_smtp": auth.user_has_permission(user, "smtp:manage"),
        "can_email_results": (
            smtp_settings.smtp_configured() and auth.user_has_permission(user, "results:inspect")
        ),
        "can_select_result_images": (
            auth.user_has_permission(user, "roboflow:manage")
            or auth.user_has_permission(user, "results:inspect")
        ),
        "can_view_result_owners": auth.can_view_all_results(user),
        "can_view_model_owners": auth.user_has_permission(user, "models:manage"),
    }
    context.update(extra_context)
    return context
