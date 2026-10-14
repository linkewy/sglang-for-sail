import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

RUNNER = Path(__file__).resolve().parents[2] / "run_suite.py"


class TestPPUCISuite(CustomTestCase):
    def test_empty_ppu_suite_fails(self):
        for disabled in (False, True):
            with self.subTest(disabled=disabled), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                runner = root / "run_suite.py"
                runner.write_text(RUNNER.read_text())
                if disabled:
                    registered = root / "registered"
                    registered.mkdir()
                    (registered / "test_disabled.py").write_text(
                        'register_ppu_ci(est_time=1, suite="stage-a-test-1-gpu-ppu", '
                        'disabled="test fixture")\n'
                    )
                result = subprocess.run(
                    [
                        sys.executable,
                        str(runner),
                        "--hw",
                        "ppu",
                        "--suite",
                        "stage-a-test-1-gpu-ppu",
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(
                    "PPU suite requires at least one enabled test", result.stderr
                )


if __name__ == "__main__":
    unittest.main()
