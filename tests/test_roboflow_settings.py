import json
import sqlite3
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import auth


class RoboflowSettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        temp_path = Path(self.temp_dir.name)
        self.database_path = temp_path / "tilletia.sqlite3"
        self.settings_path = temp_path / "roboflow.json"
        self.master_key_path = temp_path / "roboflow-master-key"
        self.auth_paths = patch.multiple(
            auth,
            DB_PATH=str(self.database_path),
            ROBOFLOW_SETTINGS_PATH=str(self.settings_path),
            ROBOFLOW_MASTER_KEY_PATH=str(self.master_key_path),
        )
        self.auth_paths.start()
        auth._roboflow_memory_api_token = ""
        auth.init_auth_storage()

    def tearDown(self):
        auth._roboflow_memory_api_token = ""
        self.auth_paths.stop()
        self.temp_dir.cleanup()

    def _install_master_key(self, key=b"K" * 32):
        self.master_key_path.write_bytes(key)
        self.master_key_path.chmod(0o400)

    def _database_token_fields(self):
        connection = sqlite3.connect(self.database_path)
        try:
            return connection.execute(
                """
                SELECT api_token_nonce, api_token_ciphertext, key_version
                FROM roboflow_settings
                WHERE id = 1
                """
            ).fetchone()
        finally:
            connection.close()

    def test_secure_storage_encrypts_token_in_sqlite_without_exposing_it(self):
        self._install_master_key()

        result = auth.update_roboflow_settings({
            "project_name": "my-project",
            "api_token": "private-token",
        })
        safe_settings = auth.get_roboflow_settings()
        internal_settings = auth.get_roboflow_settings(include_api_token=True)
        token_fields = self._database_token_fields()

        self.assertNotIn("api_token", result)
        self.assertNotIn("api_token", safe_settings)
        self.assertEqual(result["project_name"], "my-project")
        self.assertTrue(result["api_token_configured"])
        self.assertTrue(result["api_token_usable"])
        self.assertTrue(result["secure_storage_available"])
        self.assertEqual(result["secure_storage_status"], "available")
        self.assertEqual(result["api_token_storage"], "encrypted")
        self.assertTrue(result["api_token_persistent"])
        self.assertEqual(internal_settings["api_token"], "private-token")
        self.assertEqual(len(token_fields[0]), auth.ROBOFLOW_TOKEN_NONCE_BYTES)
        self.assertNotEqual(bytes(token_fields[1]), b"private-token")
        self.assertEqual(token_fields[2], auth.ROBOFLOW_KEY_VERSION)
        self.assertNotIn(b"private-token", self.database_path.read_bytes())
        self.assertFalse(self.settings_path.exists())

    def test_encrypted_token_can_be_decrypted_after_restart_with_same_key(self):
        self._install_master_key()
        auth.update_roboflow_settings({
            "project_name": "restart-project",
            "api_token": "restart-token",
        })

        auth._roboflow_memory_api_token = ""
        auth.init_auth_storage()

        settings = auth.get_roboflow_settings(include_api_token=True)
        self.assertEqual(settings["project_name"], "restart-project")
        self.assertEqual(settings["api_token"], "restart-token")
        self.assertEqual(settings["api_token_storage"], "encrypted")
        self.assertTrue(settings["api_token_persistent"])

    def test_missing_key_defaults_to_memory_and_token_is_lost_after_restart(self):
        result = auth.update_roboflow_settings({
            "project_name": "memory-project",
            "api_token": "memory-token",
        })

        self.assertEqual(result["secure_storage_status"], "unavailable")
        self.assertFalse(result["secure_storage_available"])
        self.assertEqual(result["api_token_storage"], "memory")
        self.assertFalse(result["api_token_persistent"])
        self.assertEqual(
            auth.get_roboflow_settings(include_api_token=True)["api_token"],
            "memory-token",
        )
        self.assertFalse(self.settings_path.exists())

        auth._roboflow_memory_api_token = ""
        auth.init_auth_storage()

        restarted_settings = auth.get_roboflow_settings(include_api_token=True)
        self.assertEqual(restarted_settings["project_name"], "memory-project")
        self.assertEqual(restarted_settings["api_token"], "")
        self.assertFalse(restarted_settings["api_token_configured"])
        self.assertEqual(restarted_settings["api_token_storage"], "none")

    def test_declining_plaintext_file_storage_keeps_token_in_memory(self):
        result = auth.update_roboflow_settings({
            "project_name": "memory-project",
            "api_token": "memory-token",
            "allow_insecure_file_storage": False,
        })

        self.assertEqual(result["api_token_storage"], "memory")
        self.assertFalse(result["api_token_persistent"])
        self.assertFalse(self.settings_path.exists())

    def test_explicit_confirmation_persists_plaintext_file_with_mode_0600(self):
        result = auth.update_roboflow_settings({
            "project_name": "fallback-project",
            "api_token": "fallback-token",
            "allow_insecure_file_storage": True,
        })

        self.assertEqual(result["api_token_storage"], "plaintext_file")
        self.assertTrue(result["api_token_persistent"])
        self.assertEqual(stat.S_IMODE(self.settings_path.stat().st_mode), 0o600)
        self.assertEqual(
            json.loads(self.settings_path.read_text(encoding="utf-8")),
            {
                "schema_version": 1,
                "project_name": "fallback-project",
                "api_token": "fallback-token",
                "plaintext_storage_confirmed": True,
            },
        )

        auth._roboflow_memory_api_token = ""
        auth.init_auth_storage()

        restarted_settings = auth.get_roboflow_settings(include_api_token=True)
        self.assertEqual(restarted_settings["api_token"], "fallback-token")
        self.assertEqual(restarted_settings["api_token_storage"], "plaintext_file")

    def test_non_boolean_plaintext_confirmation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "confirmation must be a boolean"):
            auth.update_roboflow_settings({
                "project_name": "my-project",
                "api_token": "private-token",
                "allow_insecure_file_storage": "true",
            })

        self.assertFalse(self.settings_path.exists())

    def test_secure_storage_wins_over_plaintext_confirmation(self):
        auth.update_roboflow_settings({
            "project_name": "my-project",
            "api_token": "private-token",
            "allow_insecure_file_storage": True,
        })
        self.assertTrue(self.settings_path.exists())
        self._install_master_key()

        result = auth.update_roboflow_settings({
            "project_name": "my-project",
            "api_token": "",
            "allow_insecure_file_storage": True,
        })

        self.assertEqual(result["api_token_storage"], "encrypted")
        self.assertTrue(result["secure_storage_available"])
        self.assertFalse(self.settings_path.exists())
        self.assertEqual(
            auth.get_roboflow_settings(include_api_token=True)["api_token"],
            "private-token",
        )

    def test_legacy_plaintext_file_is_migrated_to_encrypted_storage(self):
        self._install_master_key()
        self.settings_path.write_text(
            json.dumps({
                "project_name": "legacy-project",
                "api_token": "legacy-token",
            }),
            encoding="utf-8",
        )

        auth.init_auth_storage()

        settings = auth.get_roboflow_settings(include_api_token=True)
        self.assertEqual(settings["api_token"], "legacy-token")
        self.assertEqual(settings["api_token_storage"], "encrypted")
        self.assertFalse(self.settings_path.exists())
        self.assertNotIn(b"legacy-token", self.database_path.read_bytes())

    def test_unconfirmed_legacy_plaintext_file_moves_to_memory_without_key(self):
        self.settings_path.write_text(
            json.dumps({
                "project_name": "legacy-project",
                "api_token": "legacy-token",
            }),
            encoding="utf-8",
        )

        auth.init_auth_storage()

        settings = auth.get_roboflow_settings(include_api_token=True)
        self.assertEqual(settings["api_token"], "legacy-token")
        self.assertEqual(settings["api_token_storage"], "memory")
        self.assertFalse(settings["api_token_persistent"])
        self.assertFalse(self.settings_path.exists())

    def test_malformed_legacy_file_does_not_block_replacement(self):
        self.settings_path.write_text("{not-json", encoding="utf-8")

        auth.init_auth_storage()

        unavailable = auth.get_roboflow_settings()
        self.assertEqual(unavailable["api_token_storage"], "plaintext_file_unavailable")
        self.assertFalse(unavailable["api_token_usable"])

        recovered = auth.update_roboflow_settings({
            "project_name": "recovered-project",
            "api_token": "replacement-token",
            "allow_insecure_file_storage": False,
        })
        self.assertEqual(recovered["api_token_storage"], "memory")
        self.assertFalse(self.settings_path.exists())

    def test_blank_token_preserves_existing_token(self):
        self._install_master_key()
        auth.update_roboflow_settings({
            "project_name": "first-project",
            "api_token": "private-token",
        })
        auth.update_roboflow_settings({
            "project_name": "second-project",
            "api_token": "",
        })

        settings = auth.get_roboflow_settings(include_api_token=True)
        self.assertEqual(settings["project_name"], "second-project")
        self.assertEqual(settings["api_token"], "private-token")
        self.assertEqual(settings["api_token_storage"], "encrypted")

    def test_workspace_project_path_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "project ID"):
            auth.update_roboflow_settings({
                "project_name": "workspace/project",
                "api_token": "private-token",
            })

    def test_non_string_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "project name must be a string"):
            auth.update_roboflow_settings({
                "project_name": ["my-project"],
                "api_token": "private-token",
            })
        with self.assertRaisesRegex(ValueError, "API token must be a string"):
            auth.update_roboflow_settings({
                "project_name": "my-project",
                "api_token": {"token": "private-token"},
            })


if __name__ == "__main__":
    unittest.main()
