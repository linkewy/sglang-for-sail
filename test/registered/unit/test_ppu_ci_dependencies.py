import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/ci/ppu/check_dependencies.py"


class TestPPUCIDependencies(CustomTestCase):
    def check_snapshots(self, before, after):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("before.json", "after.json")]
            for path, snapshot in zip(paths, (before, after)):
                path.write_text(json.dumps(snapshot))
            return subprocess.run(
                [sys.executable, str(SCRIPT), "check", *map(str, paths)],
                capture_output=True,
                text=True,
            )

    def test_only_built_packages_can_change(self):
        before = {
            "torch": {"version": "2.13.0", "record": "compatible"},
            "triton": {"version": "3.7.0", "record": "vendor"},
        }
        after = {
            **before,
            "sglang": {"version": "0.5.18", "record": "built"},
            "sglang-kernel": {"version": "0.4.9", "record": "built"},
        }
        result = self.check_snapshots(before, after)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_sailify_can_be_added_for_runtime_jit(self):
        before = {"torch": {"version": "2.13.0", "record": "compatible"}}
        after = {**before, "sailify": {"version": "1.0.0", "record": "converter"}}
        result = self.check_snapshots(before, after)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_runtime_torch_cannot_be_replaced_by_build_torch(self):
        before = {"torch": {"version": "2.13.0", "record": "compatible"}}
        after = {"torch": {"version": "2.13.0", "record": "cuda-free"}}
        result = self.check_snapshots(before, after)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("torch", result.stderr)

    def test_dependency_changes_are_rejected(self):
        before = {"triton": {"version": "3.7.0", "record": "vendor"}}
        for after in (
            {},
            {"triton": {"version": "3.8.0", "record": "vendor"}},
            {"triton": {"version": "3.7.0", "record": "upstream"}},
            {**before, "nvidia-cublas-cu13": {"version": "13", "record": "new"}},
        ):
            with self.subTest(after=after):
                result = self.check_snapshots(before, after)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Unexpected dependency changes", result.stderr)

    def test_existing_sailify_cannot_be_replaced(self):
        before = {"sailify": {"version": "1.0", "record": "vendor"}}
        for after in (
            {},
            {"sailify": {"version": "1.1", "record": "vendor"}},
            {"sailify": {"version": "1.0", "record": "different"}},
        ):
            with self.subTest(after=after):
                result = self.check_snapshots(before, after)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("sailify", result.stderr)

    def test_snapshot_uses_first_distribution_on_python_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sites = [root / name for name in ("build", "runtime")]
            for site in sites:
                info = site / "ppu_ci_snapshot_probe-1.0.dist-info"
                info.mkdir(parents=True)
                (info / "METADATA").write_text(
                    "Name: ppu-ci-snapshot-probe\nVersion: 1.0\n"
                )
                (info / "RECORD").write_text(site.name)
            output = root / "snapshot.json"
            for order in (sites, sites[::-1]):
                with self.subTest(first=order[0].name):
                    result = subprocess.run(
                        [sys.executable, str(SCRIPT), "snapshot", str(output)],
                        env={
                            **os.environ,
                            "PYTHONPATH": os.pathsep.join(map(str, order)),
                        },
                        capture_output=True,
                        text=True,
                    )
                    self.assertEqual(result.returncode, 0, result.stderr)
                    snapshot = json.loads(output.read_text())
                    self.assertEqual(
                        snapshot["ppu-ci-snapshot-probe"],
                        {
                            "version": "1.0",
                            "record": hashlib.sha256(
                                order[0].name.encode()
                            ).hexdigest(),
                        },
                    )


if __name__ == "__main__":
    unittest.main()
