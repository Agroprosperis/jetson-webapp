import sys
import tempfile
import unittest
from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from result_ids import generate_unique_result_run_id


class ResultRunIdTests(unittest.TestCase):
    def test_returns_original_id_when_no_conflict_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(
                generate_unique_result_run_id("manual-analysis", temp_dir, now_ms=1234567890123),
                "manual-analysis",
            )

    def test_appends_millisecond_stamp_when_directory_already_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "manual-analysis").mkdir()

            self.assertEqual(
                generate_unique_result_run_id("manual-analysis", temp_dir, now_ms=1234567890123),
                "manual-analysis-1234567890123",
            )

    def test_adds_counter_when_timestamped_id_already_exists(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            Path(temp_dir, "manual-analysis").mkdir()
            Path(temp_dir, "manual-analysis-1234567890123").mkdir()

            self.assertEqual(
                generate_unique_result_run_id("manual-analysis", temp_dir, now_ms=1234567890123),
                "manual-analysis-1234567890123-1",
            )


if __name__ == "__main__":
    unittest.main()
