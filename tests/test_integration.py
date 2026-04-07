import json
from pathlib import Path
import shutil
import unittest
import urllib.error
import urllib.request
import uuid


BASE_URL = "http://127.0.0.1:8000"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin"
TEST_USER_PASSWORD = "TestUser_Str0ng!Pass_2026"
TEST_USER_NEW_PASSWORD = "TestUser_Str0ng!Pass_2026_Changed"
OUTPUT_HQ_DIR = Path(__file__).resolve().parents[1] / "data" / "output_hq"


def request(method: str, path: str, token: str | None = None, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(f"{BASE_URL}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=5) as response:
        return response.status, response.headers, response.read().decode("utf-8")


def login(username: str, password: str):
    status, _, body = request("POST", "/auth/login", payload={"username": username, "password": password})
    return status, json.loads(body)


def change_password(
    username: str,
    current_password: str,
    new_password: str,
    confirm_password: str | None = None,
):
    status, _, body = request(
        "POST",
        "/auth/change-password",
        payload={
            "username": username,
            "current_password": current_password,
            "new_password": new_password,
            "confirm_password": new_password if confirm_password is None else confirm_password,
        },
    )
    return status, json.loads(body)


def admin_tokens():
    status, data = login(ADMIN_USERNAME, ADMIN_PASSWORD)
    if status != 200:
        raise AssertionError("admin/admin login must work for integration tests")
    return data


def create_user(role: str = "user", prefix: str = "test_user"):
    username = f"{prefix}_{uuid.uuid4().hex[:8]}"
    status, _, _ = request(
        "POST",
        "/users",
        payload={"username": username, "password": TEST_USER_PASSWORD, "roles": [role]},
        token=admin_tokens()["access_token"],
    )
    if status != 201:
        raise AssertionError("admin must be able to create users with roles")
    return username


def activate_user(username: str, initial_password: str = TEST_USER_PASSWORD, new_password: str = TEST_USER_NEW_PASSWORD):
    status, _ = change_password(username, initial_password, new_password)
    if status != 200:
        raise AssertionError("created user must be able to change the initial password")
    status, data = login(username, new_password)
    if status != 200:
        raise AssertionError("created user must be able to log in with the changed password")
    return data


def user_token(role: str = "user"):
    username = create_user(role=role)
    return activate_user(username)["access_token"]


class AuthIntegrationTests(unittest.TestCase):
    def test_login_success(self):
        status, data = login("admin", "admin")
        self.assertEqual(status, 200)
        self.assertIn("access_token", data)
        self.assertIn("refresh_token", data)

    def test_login_failure(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            login("admin", "wrong-password")
        self.assertEqual(ctx.exception.code, 401)

    def test_token_refresh(self):
        _, login_data = login("admin", "admin")
        status, _, body = request(
            "POST",
            "/auth/refresh",
            payload={"refresh_token": login_data["refresh_token"]},
        )
        self.assertEqual(status, 200)
        self.assertIn("access_token", json.loads(body))

    def test_admin_can_create_user_with_role(self):
        self.assertTrue(create_user().startswith("test_user_"))


class ForcedPasswordChangeTests(unittest.TestCase):
    def test_login_page_has_confirm_password_field(self):
        status, _, body = request("GET", "/login")
        self.assertEqual(status, 200)
        self.assertIn('id="confirmLoginPassword"', body)

    def test_admin_created_user_must_change_password_on_first_login(self):
        username = create_user()
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("POST", "/auth/login", payload={"username": username, "password": TEST_USER_PASSWORD})
        self.assertEqual(ctx.exception.code, 403)
        self.assertTrue(json.loads(ctx.exception.read().decode("utf-8"))["password_change_required"])

    def test_admin_created_user_can_change_initial_password(self):
        username = create_user()
        status, data = change_password(username, TEST_USER_PASSWORD, TEST_USER_NEW_PASSWORD)
        self.assertEqual(status, 200)
        self.assertTrue(data["success"])

    def test_admin_created_user_cannot_change_initial_password_with_mismatch(self):
        username = create_user()
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            change_password(username, TEST_USER_PASSWORD, TEST_USER_NEW_PASSWORD, "different-password")
        self.assertEqual(ctx.exception.code, 400)

    def test_user_can_log_in_after_initial_password_change(self):
        username = create_user()
        status, _ = change_password(username, TEST_USER_PASSWORD, TEST_USER_NEW_PASSWORD)
        self.assertEqual(status, 200)
        login_status, login_data = login(username, TEST_USER_NEW_PASSWORD)
        self.assertEqual(login_status, 200)
        self.assertIn("access_token", login_data)


class AdminCreatedAdminPermissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.username = create_user(role="admin", prefix="test_admin")
        cls.login_data = activate_user(cls.username)
        cls.login_status = 200

    def setUp(self):
        self.status, self.headers, self.body = request("GET", "/", token=self.login_data["access_token"])

    def test_admin_can_create_test_admin_user(self):
        self.assertTrue(self.username.startswith("test_admin_"))

    def test_created_admin_can_log_in(self):
        self.assertEqual(self.login_status, 200)
        self.assertIn("access_token", self.login_data)
        self.assertIn("refresh_token", self.login_data)

    def test_created_admin_can_create_users(self):
        status, _, _ = request(
            "POST",
            "/users",
            token=self.login_data["access_token"],
            payload={"username": f"test_user_{uuid.uuid4().hex[:8]}", "password": TEST_USER_PASSWORD, "roles": ["user"]},
        )
        self.assertEqual(status, 201)

    def test_created_admin_can_access_main_panel(self):
        self.assertEqual(self.status, 200)

    def test_created_admin_can_access_models_page(self):
        status, _, _ = request("GET", "/models", token=self.login_data["access_token"])
        self.assertEqual(status, 200)

    def test_created_admin_sees_models_page_tensorrt_info(self):
        status, _, body = request("GET", "/models", token=self.login_data["access_token"])
        self.assertEqual(status, 200)
        self.assertIn('id="tensorrtInfo"', body)

    def test_created_admin_sees_model_upload_type_selector(self):
        status, _, body = request("GET", "/models", token=self.login_data["access_token"])
        self.assertEqual(status, 200)
        self.assertIn('id="uploadModelType"', body)

    def test_created_admin_sees_model_upload_button(self):
        status, _, body = request("GET", "/models", token=self.login_data["access_token"])
        self.assertEqual(status, 200)
        self.assertIn('id="uploadModelBtn"', body)

    def test_created_admin_sees_model_upload_input(self):
        status, _, body = request("GET", "/models", token=self.login_data["access_token"])
        self.assertEqual(status, 200)
        self.assertIn('id="uploadModelInput"', body)

    def test_created_admin_sees_model_upload_status(self):
        status, _, body = request("GET", "/models", token=self.login_data["access_token"])
        self.assertEqual(status, 200)
        self.assertIn('id="uploadModelStatus"', body)

    def test_created_admin_sees_compile_logs_button(self):
        status, _, body = request("GET", "/models", token=self.login_data["access_token"])
        self.assertEqual(status, 200)
        self.assertIn('id="openCompileLogs"', body)

    def test_created_admin_sees_models_table(self):
        status, _, body = request("GET", "/models", token=self.login_data["access_token"])
        self.assertEqual(status, 200)
        self.assertIn('id="modelsTable"', body)

    def test_created_admin_sees_models_link(self):
        self.assertIn('href="/models"', self.body)

    def test_created_admin_sees_users_link(self):
        self.assertIn('href="/users"', self.body)

    def test_created_admin_sees_snapshot_button(self):
        self.assertIn("id='snapshotBtn'", self.body)

    def test_created_admin_sees_settings_link(self):
        self.assertIn('href="/settings"', self.body)

    def test_created_admin_sees_logout_button(self):
        self.assertIn('id="logoutBtn"', self.body)

    def test_created_admin_can_edit_analysis_number(self):
        self.assertNotRegex(self.body, r"id=['\"]analysisNum['\"][^>]*(disabled|hidden)")

    def test_created_admin_can_change_source_type(self):
        self.assertNotRegex(self.body, r"id=['\"]sourceType['\"][^>]*(disabled|hidden)")

    def test_created_admin_can_change_camera_device(self):
        self.assertNotRegex(self.body, r"id=['\"]camDevice['\"][^>]*(disabled|hidden)")

    def test_created_admin_can_change_camera_mode(self):
        self.assertNotRegex(self.body, r"id=['\"]camMode['\"][^>]*(disabled|hidden)")

    def test_created_admin_can_refresh_cameras(self):
        self.assertNotRegex(self.body, r"id=['\"]refreshCams['\"][^>]*(disabled|hidden)")

    def test_created_admin_can_change_model(self):
        self.assertNotRegex(self.body, r"id=['\"]modelSelect['\"][^>]*(disabled|hidden)")

    def test_created_admin_can_change_confidence(self):
        self.assertNotRegex(self.body, r"id=['\"]confThreshold['\"][^>]*(disabled|hidden)")

    def test_created_admin_can_change_grid_count(self):
        self.assertNotRegex(self.body, r"id=['\"]gridCountEnabled['\"][^>]*(disabled|hidden)")

    def test_created_admin_can_change_grid_debug(self):
        self.assertNotRegex(self.body, r"id=['\"]gridDebugEnabled['\"][^>]*(disabled|hidden)")

    def test_created_admin_can_change_grid_score_threshold(self):
        self.assertNotRegex(self.body, r"id=['\"]gridScoreThreshold['\"][^>]*(disabled|hidden)")

    def test_created_admin_can_refresh_cameras_via_http(self):
        status, _, _ = request("GET", "/api/cameras", token=self.login_data["access_token"])
        self.assertEqual(status, 200)

    def test_created_admin_can_list_models_via_http(self):
        status, _, body = request("GET", "/api/models", token=self.login_data["access_token"])
        self.assertEqual(status, 200)
        self.assertIn("models", json.loads(body))

    def test_created_admin_can_read_model_catalog_via_http(self):
        status, _, body = request("GET", "/api/model-catalog", token=self.login_data["access_token"])
        self.assertEqual(status, 200)
        self.assertIn("models", json.loads(body))

    def test_created_admin_can_read_model_compile_jobs_via_http(self):
        status, _, body = request("GET", "/api/model-compile-jobs", token=self.login_data["access_token"])
        self.assertEqual(status, 200)
        self.assertIn("jobs", json.loads(body))

class PermissionIntegrationTests(unittest.TestCase):
    def test_401_without_token(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("GET", "/api/status")
        self.assertEqual(ctx.exception.code, 401)

    def test_403_with_wrong_role(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("GET", "/models", token=user_token())
        self.assertEqual(ctx.exception.code, 403)

    def test_user_cannot_access_users_page(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("GET", "/users", token=user_token())
        self.assertEqual(ctx.exception.code, 403)

    def test_user_cannot_delete_results(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("DELETE", "/api/results/smoke-test-result", token=user_token())
        self.assertEqual(ctx.exception.code, 403)

    def test_snapshot_requires_auth(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("POST", "/api/snapshot")
        self.assertEqual(ctx.exception.code, 401)

    def test_user_can_reach_snapshot_api(self):
        req = urllib.request.Request(
            f"{BASE_URL}/api/snapshot",
            headers={"Authorization": f"Bearer {user_token()}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                status = response.status
        except urllib.error.HTTPError as exc:
            status = exc.code

        self.assertIn(status, (200, 400, 504))

    def test_user_can_access_rest(self):
        status, _, _ = request("GET", "/api/status", token=user_token())
        self.assertEqual(status, 200)

    def test_swagger_spec_is_public(self):
        status, headers, body = request("GET", "/apispec_1.json")
        self.assertEqual(status, 200)
        self.assertIn("application/json", headers.get("Content-Type", ""))
        self.assertIn("/api/config", json.loads(body)["paths"])

    def test_swagger_ui_is_public(self):
        status, headers, body = request("GET", "/api/docs/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers.get("Content-Type", ""))
        self.assertIn("swagger", body.lower())

    def test_admin_can_access_main_panel_html(self):
        status, headers, body = request("GET", "/", token=admin_tokens()["access_token"])
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers.get("Content-Type", ""))
        self.assertIn("<html", body.lower())


class UserMainPanelPermissionTests(unittest.TestCase):
    def setUp(self):
        self.status, self.headers, self.body = request("GET", "/", token=user_token())

    def test_user_can_open_main_panel(self):
        self.assertEqual(self.status, 200)

    def test_main_panel_returns_html(self):
        self.assertIn("text/html", self.headers.get("Content-Type", ""))

    def test_user_sees_run_button(self):
        self.assertIn("id='run'", self.body)

    def test_user_sees_stop_button(self):
        self.assertIn("id='stop'", self.body)

    def test_user_sees_snapshot_button(self):
        self.assertIn("id='snapshotBtn'", self.body)

    def test_user_sees_results_link(self):
        self.assertIn('href="/results"', self.body)

    def test_user_sees_settings_link(self):
        self.assertIn('href="/settings"', self.body)

    def test_user_sees_logout_button(self):
        self.assertIn('id="logoutBtn"', self.body)

    def test_user_does_not_see_models_link(self):
        self.assertNotIn('href="/models"', self.body)

    def test_user_does_not_see_users_link(self):
        self.assertNotIn('href="/users"', self.body)

    def test_user_cannot_edit_analysis_number(self):
        self.assertRegex(self.body, r"id=['\"]analysisNum['\"][^>]*placeholder=['\"]Auto['\"][^>]*(disabled[^>]*hidden|hidden[^>]*disabled)")

    def test_user_cannot_change_source_type(self):
        self.assertRegex(self.body, r"id=['\"]sourceType['\"][^>]*(disabled[^>]*hidden|hidden[^>]*disabled)")

    def test_user_cannot_change_camera_device(self):
        self.assertRegex(self.body, r"id=['\"]camDevice['\"][^>]*(disabled[^>]*hidden|hidden[^>]*disabled)")

    def test_user_cannot_change_camera_mode(self):
        self.assertRegex(self.body, r"id=['\"]camMode['\"][^>]*(disabled[^>]*hidden|hidden[^>]*disabled)")

    def test_user_cannot_refresh_cameras(self):
        self.assertRegex(self.body, r"id=['\"]refreshCams['\"][^>]*(disabled[^>]*hidden|hidden[^>]*disabled)")

    def test_user_cannot_change_model(self):
        self.assertRegex(self.body, r"id=['\"]modelSelect['\"][^>]*(disabled[^>]*hidden|hidden[^>]*disabled)")

    def test_user_sees_default_confidence_locked(self):
        self.assertRegex(self.body, r"id=['\"]confThreshold['\"][^>]*value=['\"]0\.75['\"][^>]*(disabled[^>]*hidden|hidden[^>]*disabled)")

    def test_user_sees_grid_count_enabled_locked(self):
        self.assertRegex(self.body, r"id=['\"]gridCountEnabled['\"][^>]*checked[^>]*(disabled[^>]*hidden|hidden[^>]*disabled)")

    def test_user_cannot_change_grid_debug(self):
        self.assertRegex(self.body, r"id=['\"]gridDebugEnabled['\"][^>]*(disabled[^>]*hidden|hidden[^>]*disabled)")

    def test_user_sees_default_grid_score_threshold_locked(self):
        self.assertRegex(self.body, r"id=['\"]gridScoreThreshold['\"][^>]*value=['\"]0\.30['\"][^>]*(disabled[^>]*hidden|hidden[^>]*disabled)")


class UserDirectModificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.token = user_token()

    def test_user_cannot_override_analysis_number_via_http(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("POST", "/api/start", token=self.token, payload={"analysis_number": "manual-id"})
        self.assertEqual(ctx.exception.code, 403)

    def test_user_cannot_override_source_type_via_http(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("POST", "/api/start", token=self.token, payload={"source_type": "file", "video": "/tmp/x.mp4"})
        self.assertEqual(ctx.exception.code, 403)

    def test_user_cannot_override_camera_device_via_http(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("POST", "/api/start", token=self.token, payload={"source_type": "camera", "device": "/dev/video9"})
        self.assertEqual(ctx.exception.code, 403)

    def test_user_cannot_override_camera_mode_via_http(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request(
                "POST",
                "/api/start",
                token=self.token,
                payload={"source_type": "camera", "width": 1920, "height": 1080, "fps": 15, "format": "YUYV"},
            )
        self.assertEqual(ctx.exception.code, 403)

    def test_user_cannot_refresh_cameras_via_http(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("GET", "/api/cameras", token=self.token)
        self.assertEqual(ctx.exception.code, 403)

    def test_user_cannot_upload_video_via_http(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("POST", "/api/upload", token=self.token, payload={})
        self.assertEqual(ctx.exception.code, 403)

    def test_user_cannot_override_model_via_http(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("POST", "/api/start", token=self.token, payload={"model_path": "/app/model/ul/custom.engine"})
        self.assertEqual(ctx.exception.code, 403)

    def test_user_cannot_override_confidence_via_http(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("POST", "/api/start", token=self.token, payload={"vis_conf": 0.25})
        self.assertEqual(ctx.exception.code, 403)

    def test_user_cannot_override_grid_count_via_http(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("PUT", "/api/grid", token=self.token, payload={"enabled": False})
        self.assertEqual(ctx.exception.code, 403)

    def test_user_cannot_override_grid_debug_via_http(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("PUT", "/api/grid", token=self.token, payload={"debug_enabled": True})
        self.assertEqual(ctx.exception.code, 403)

    def test_user_cannot_override_grid_score_threshold_via_http(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("PUT", "/api/grid", token=self.token, payload={"score_threshold": 0.10})
        self.assertEqual(ctx.exception.code, 403)


class ZDashboardSettingsValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.admin_token = admin_tokens()["access_token"]
        cls.user_token = user_token()

    def test_admin_selected_source_type_is_used_for_user(self):
        status, _, _ = request("PUT", "/api/dashboard-settings", token=self.admin_token, payload={"source_type": "file"})
        self.assertEqual(status, 200)
        status, _, body = request("GET", "/api/dashboard-settings", token=self.user_token)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["source_type"], "file")

    def test_admin_selected_camera_device_is_used_for_user(self):
        status, _, _ = request("PUT", "/api/dashboard-settings", token=self.admin_token, payload={"camera_device": "/dev/video7"})
        self.assertEqual(status, 200)
        status, _, body = request("GET", "/api/dashboard-settings", token=self.user_token)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["camera_device"], "/dev/video7")

    def test_admin_selected_camera_mode_is_used_for_user(self):
        status, _, _ = request(
            "PUT",
            "/api/dashboard-settings",
            token=self.admin_token,
            payload={"camera_mode": {"width": 1920, "height": 1080, "fps": 15, "format": "YUYV"}},
        )
        self.assertEqual(status, 200)
        status, _, body = request("GET", "/api/dashboard-settings", token=self.user_token)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["camera_mode"], {"width": 1920, "height": 1080, "fps": 15, "format": "YUYV"})

    def test_admin_selected_model_is_used_for_user(self):
        status, _, _ = request(
            "PUT",
            "/api/dashboard-settings",
            token=self.admin_token,
            payload={"model_path": "/app/model/ul/custom.engine"},
        )
        self.assertEqual(status, 200)
        status, _, body = request("GET", "/api/dashboard-settings", token=self.user_token)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["model_path"], "/app/model/ul/custom.engine")

    def test_admin_selected_confidence_is_used_for_user(self):
        status, _, _ = request("PUT", "/api/dashboard-settings", token=self.admin_token, payload={"vis_conf": 0.55})
        self.assertEqual(status, 200)
        status, _, body = request("GET", "/api/dashboard-settings", token=self.user_token)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["vis_conf"], 0.55)

    def test_admin_selected_grid_count_is_used_for_user(self):
        status, _, _ = request("PUT", "/api/dashboard-settings", token=self.admin_token, payload={"grid_count_enabled": False})
        self.assertEqual(status, 200)
        status, _, body = request("GET", "/api/dashboard-settings", token=self.user_token)
        self.assertEqual(status, 200)
        self.assertFalse(json.loads(body)["grid_count_enabled"])

    def test_admin_selected_grid_debug_is_used_for_user(self):
        status, _, _ = request("PUT", "/api/dashboard-settings", token=self.admin_token, payload={"grid_debug_enabled": True})
        self.assertEqual(status, 200)
        status, _, body = request("GET", "/api/dashboard-settings", token=self.user_token)
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["grid_debug_enabled"])

    def test_admin_selected_grid_score_threshold_is_used_for_user(self):
        status, _, _ = request("PUT", "/api/dashboard-settings", token=self.admin_token, payload={"grid_score_threshold": 0.65})
        self.assertEqual(status, 200)
        status, _, body = request("GET", "/api/dashboard-settings", token=self.user_token)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["grid_score_threshold"], 0.65)


class UserResultsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.token = user_token()

    def setUp(self):
        self.status, self.headers, self.body = request("GET", "/results", token=self.token)

    def test_user_can_open_results_page(self):
        self.assertEqual(self.status, 200)

    def test_results_page_returns_html(self):
        self.assertIn("text/html", self.headers.get("Content-Type", ""))

    def test_results_page_has_date_filter(self):
        self.assertIn('id="dateFilter"', self.body)

    def test_results_page_has_apply_filter_button(self):
        self.assertIn('id="applyFilters"', self.body)

    def test_results_page_has_clear_filter_button(self):
        self.assertIn('id="clearFilters"', self.body)

    def test_results_page_has_results_table(self):
        self.assertIn('id="resultsTable"', self.body)

    def test_results_page_does_not_have_owner_column(self):
        self.assertNotIn("<th>Owner</th>", self.body)


class AdminResultsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.token = admin_tokens()["access_token"]

    def test_admin_results_page_has_owner_column(self):
        status, _, body = request("GET", "/results", token=self.token)
        self.assertEqual(status, 200)
        self.assertIn("<th>Owner</th>", body)


class AdminUsersPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.token = admin_tokens()["access_token"]

    def setUp(self):
        self.status, self.headers, self.body = request("GET", "/users", token=self.token)

    def test_admin_can_open_users_page(self):
        self.assertEqual(self.status, 200)

    def test_users_page_returns_html(self):
        self.assertIn("text/html", self.headers.get("Content-Type", ""))

    def test_users_page_has_users_table(self):
        self.assertIn('id="usersTable"', self.body)

    def test_users_page_has_username_input(self):
        self.assertIn('id="newUsername"', self.body)

    def test_users_page_has_initial_password_input(self):
        self.assertIn('id="newPassword"', self.body)

    def test_users_page_has_role_selector(self):
        self.assertIn('id="newRole"', self.body)

    def test_users_page_has_create_user_button(self):
        self.assertIn('id="createUserBtn"', self.body)

    def test_users_page_explains_password_change_on_first_login(self):
        self.assertIn("change password on first login", self.body.lower())


class UserSettingsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.username = create_user()
        cls.token = activate_user(cls.username)["access_token"]

    def setUp(self):
        self.status, self.headers, self.body = request("GET", "/settings", token=self.token)

    def test_user_can_open_settings_page(self):
        self.assertEqual(self.status, 200)

    def test_settings_page_returns_html(self):
        self.assertIn("text/html", self.headers.get("Content-Type", ""))

    def test_settings_page_shows_read_only_username(self):
        self.assertRegex(
            self.body,
            rf"id=\"settingsUsername\"[^>]*value=\"{self.username}\"[^>]*readonly",
        )

    def test_settings_page_shows_read_only_role(self):
        self.assertRegex(
            self.body,
            r"id=\"settingsRole\"[^>]*value=\"user\"[^>]*readonly",
        )

    def test_settings_page_has_password_fields(self):
        self.assertIn('id="settingsCurrentPassword"', self.body)
        self.assertIn('id="settingsNewPassword"', self.body)
        self.assertIn('id="settingsConfirmPassword"', self.body)
        self.assertIn('id="changePasswordBtn"', self.body)


class AdminSettingsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.token = admin_tokens()["access_token"]

    def test_admin_can_open_settings_page(self):
        status, _, body = request("GET", "/settings", token=self.token)
        self.assertEqual(status, 200)
        self.assertRegex(body, r"id=\"settingsRole\"[^>]*value=\"admin\"[^>]*readonly")


class AuthenticatedPasswordChangeTests(unittest.TestCase):
    def test_user_can_change_password_after_activation(self):
        username = create_user(prefix="settings_user")
        activate_user(username)
        updated_password = f"{TEST_USER_NEW_PASSWORD}_{uuid.uuid4().hex[:8]}"

        status, data = change_password(username, TEST_USER_NEW_PASSWORD, updated_password)
        self.assertEqual(status, 200)
        self.assertTrue(data["success"])

        login_status, login_data = login(username, updated_password)
        self.assertEqual(login_status, 200)
        self.assertIn("access_token", login_data)


class ZAdminCreatedAdminOverrideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.username = create_user(role="admin", prefix="test_admin")
        cls.login_data = activate_user(cls.username)

    def test_created_admin_can_override_grid_count_via_http(self):
        status, _, _ = request("PUT", "/api/grid", token=self.login_data["access_token"], payload={"enabled": False})
        self.assertEqual(status, 200)

    def test_created_admin_can_override_grid_debug_via_http(self):
        status, _, _ = request("PUT", "/api/grid", token=self.login_data["access_token"], payload={"debug_enabled": True})
        self.assertEqual(status, 200)

    def test_created_admin_can_override_grid_score_threshold_via_http(self):
        status, _, _ = request("PUT", "/api/grid", token=self.login_data["access_token"], payload={"score_threshold": 0.65})
        self.assertEqual(status, 200)

    def test_created_admin_can_override_dashboard_settings_via_http(self):
        status, _, _ = request(
            "PUT",
            "/api/dashboard-settings",
            token=self.login_data["access_token"],
            payload={"vis_conf": 0.55, "model_path": "/app/model/ul/custom.engine"},
        )
        self.assertEqual(status, 200)


class UserResultsVisibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.username = create_user()
        cls.token = activate_user(cls.username)["access_token"]
        cls.admin_token = admin_tokens()["access_token"]
        cls.user_run = f"user-run-{uuid.uuid4().hex[:8]}"
        cls.admin_run = f"admin-run-{uuid.uuid4().hex[:8]}"

        for run_id, owner in ((cls.user_run, cls.username), (cls.admin_run, ADMIN_USERNAME)):
            run_dir = OUTPUT_HQ_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "metadata.json").write_text(json.dumps({"owner": owner}), encoding="utf-8")
            (run_dir / f"{run_id}.csv").write_text(
                "frame,analysis_number,s_value,total_unique_objects,detections\n1,test,0.0,0,\n",
                encoding="utf-8",
            )
            (run_dir / f"{run_id}.mkv").write_bytes(b"0")

    @classmethod
    def tearDownClass(cls):
        for run_id in (cls.user_run, cls.admin_run):
            shutil.rmtree(OUTPUT_HQ_DIR / run_id, ignore_errors=True)

    def test_user_sees_only_own_results(self):
        status, _, body = request("GET", "/api/results", token=self.token)
        self.assertEqual(status, 200)
        result_ids = [item["id"] for item in json.loads(body)["results"]]
        self.assertIn(self.user_run, result_ids)
        self.assertNotIn(self.admin_run, result_ids)

    def test_user_search_sees_only_own_results(self):
        status, _, body = request("GET", f"/api/results/search?analysis_id=run-", token=self.token)
        self.assertEqual(status, 200)
        result_ids = [item["analysis_id"] for item in json.loads(body)["results"]]
        self.assertIn(self.user_run, result_ids)
        self.assertNotIn(self.admin_run, result_ids)

    def test_user_can_read_last_row_for_own_result(self):
        status, _, body = request("GET", f"/api/results/{self.user_run}/last-row", token=self.token)
        self.assertEqual(status, 200)
        data = json.loads(body)
        self.assertEqual(data["analysis_id"], self.user_run)
        self.assertEqual(data["row"]["analysis_number"], "test")

    def test_user_cannot_read_last_row_for_foreign_result(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("GET", f"/api/results/{self.admin_run}/last-row", token=self.token)
        self.assertEqual(ctx.exception.code, 403)

    def test_user_can_download_own_result_file(self):
        status, headers, body = request("GET", f"/download/{self.user_run}/{self.user_run}.csv", token=self.token)
        self.assertEqual(status, 200)
        self.assertIn("text/csv", headers.get("Content-Type", ""))
        self.assertIn("frame,analysis_number", body)

    def test_user_cannot_download_foreign_result_file(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("GET", f"/download/{self.admin_run}/{self.admin_run}.csv", token=self.token)
        self.assertEqual(ctx.exception.code, 403)

    def test_user_can_download_own_result_zip(self):
        req = urllib.request.Request(
            f"{BASE_URL}/api/results/{self.user_run}/download",
            headers={"Authorization": f"Bearer {self.token}"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            self.assertEqual(response.status, 200)
            self.assertIn("application/zip", response.headers.get("Content-Type", ""))
            self.assertTrue(response.read().startswith(b"PK"))

    def test_user_cannot_download_foreign_result_zip(self):
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            request("GET", f"/api/results/{self.admin_run}/download", token=self.token)
        self.assertEqual(ctx.exception.code, 403)

    def test_admin_results_api_exposes_owner_username(self):
        status, _, body = request("GET", "/api/results", token=self.admin_token)
        self.assertEqual(status, 200)
        results = {item["id"]: item for item in json.loads(body)["results"]}
        self.assertEqual(results[self.user_run]["owner_username"], self.username)
        self.assertEqual(results[self.admin_run]["owner_username"], ADMIN_USERNAME)


class AdminModelsOwnerColumnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.token = admin_tokens()["access_token"]

    def test_models_page_has_owner_column(self):
        status, _, body = request("GET", "/models", token=self.token)
        self.assertEqual(status, 200)
        self.assertIn("<th>Owner</th>", body)

    def test_model_catalog_exposes_owner_username(self):
        status, _, body = request("GET", "/api/model-catalog", token=self.token)
        self.assertEqual(status, 200)
        models = json.loads(body)["models"]
        self.assertTrue(models)
        self.assertIn("owner_username", models[0])


if __name__ == "__main__":
    unittest.main()
