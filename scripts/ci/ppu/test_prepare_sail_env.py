import importlib.util
import tempfile
import unittest
from pathlib import Path

TOOL = Path(__file__).with_name("prepare_sail_env.py")


class TestPrepareEnvironment(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TOOL.is_file(), "environment helper is missing")
        spec = importlib.util.spec_from_file_location("prepare", TOOL)
        self.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tool)
        self.protected = {
            "distributions": {"torch": {"version": "2.13.0", "path": "/sdk"}},
            "modules": {
                "torch": {
                    "distribution": "torch",
                    "version": "2.13.0",
                    "path": "/sdk/torch/__init__.py",
                }
            },
        }

    def entry(self, name="numpy", url="https://example.org/numpy-1.0-py3-none-any.whl"):
        return {
            "metadata": {"name": name, "version": "1.0"},
            "download_info": {
                "url": url,
                "archive_info": {"hashes": {"sha256": "a" * 64}},
            },
        }

    def test_accept_binary_plan(self):
        result = self.tool.validate_plan(
            {"install": [self.entry()]}, self.protected, {}
        )
        self.assertEqual(result[0]["name"], "numpy")

    def test_reject_protected_even_same_version(self):
        for name in (
            "torch",
            "triton",
            "torch_for_sail",
            "triton-for-sail",
            "nvidia-cuda-runtime-cu13",
        ):
            item = self.entry(name)
            item["metadata"]["version"] = "2.13.0"
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.tool.validate_plan({"install": [item]}, self.protected, {})

    def test_reject_unsafe_downloads(self):
        for url in (
            "https://user:password@example.org/a.whl",
            "https://example.org/a.tar.gz",
            "file:///tmp/a.whl",
            "https://example.org/a.whl?token=secret",
            "http://example.org/a.whl",
        ):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.tool.validate_plan(
                    {"install": [self.entry(url=url)]}, self.protected, {}
                )

    def test_reject_project_from_index(self):
        with self.assertRaises(ValueError):
            self.tool.validate_plan(
                {"install": [self.entry("sglang")]}, self.protected, {}
            )

    def test_reject_path_drift(self):
        after = {
            **self.protected,
            "modules": {"torch": {"path": "/venv/torch/__init__.py"}},
        }
        with self.assertRaises(ValueError):
            self.tool.compare_protected(self.protected, after)

    def test_reject_source_and_system_shadowing(self):
        for path in (
            "/checkout/python/sglang/__init__.py",
            "/system/site-packages/sglang/__init__.py",
        ):
            with self.assertRaises(ValueError):
                self.tool.require_venv_path(Path(path), Path("/venv"))

    def test_locked_plan_rejects_provider_and_duplicates_before_download(self):
        self.assertTrue(
            hasattr(self.tool, "validate_locked"), "locked plan preflight is missing"
        )
        record = {
            "name": "numpy",
            "version": "1.0",
            "url": "https://example.org/numpy-1.0-py3-none-any.whl",
            "filename": "numpy-1.0-py3-none-any.whl",
            "sha256": "a" * 64,
        }
        protected = {"distributions": {"vendor-torch-provider": {"version": "1.0"}}}
        self.tool.validate_locked([record], protected)
        for records in (
            [record, record],
            [dict(record, name="vendor-torch-provider")],
            [dict(record, name="Torch")],
            [dict(record, sha256="bad")],
        ):
            with self.subTest(records=records), self.assertRaises(ValueError):
                self.tool.validate_locked(records, protected)

    def test_dependency_archive_paths(self):
        import zipfile

        for member in (
            "../outside",
            "/absolute",
            "numpy\\escape",
            "numpy-1.0.data/purelib/torch/__init__.py",
            "numpy-1.0.data/platlib/triton/__init__.py",
            "torch.py",
            "triton.cpython-312-x86_64-linux-gnu.so",
        ):
            with self.subTest(member=member), tempfile.TemporaryDirectory() as temp:
                path = Path(temp) / "numpy-1.0-py3-none-any.whl"
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr(
                        "numpy-1.0.dist-info/METADATA", "Name: numpy\nVersion: 1.0\n"
                    )
                    archive.writestr(member, "bad")
                with self.assertRaises(ValueError):
                    self.tool.verify_download(
                        path,
                        {
                            "sha256": self.tool.sha256(path),
                            "name": "numpy",
                            "version": "1.0",
                        },
                    )

    def test_dependency_symlink(self):
        import stat
        import zipfile

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "numpy-1.0-py3-none-any.whl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "numpy-1.0.dist-info/METADATA", "Name: numpy\nVersion: 1.0\n"
                )
                link = zipfile.ZipInfo("numpy/link")
                link.create_system = 3
                link.external_attr = (stat.S_IFLNK | 0o777) << 16
                archive.writestr(link, "../../torch")
            with self.assertRaises(ValueError):
                self.tool.verify_download(
                    path,
                    {
                        "sha256": self.tool.sha256(path),
                        "name": "numpy",
                        "version": "1.0",
                    },
                )

    def test_install_preflights_entire_lock(self):
        from unittest.mock import patch

        record = {
            "name": "numpy",
            "version": "1.0",
            "url": "https://example.org/numpy-1.0-py3-none-any.whl",
            "filename": "numpy-1.0-py3-none-any.whl",
            "sha256": "a" * 64,
        }
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(self.tool, "urlopen") as download,
        ):
            self.assertIn(
                "protected",
                __import__("inspect").signature(self.tool.install_locked).parameters,
            )
            with self.assertRaises(ValueError):
                self.tool.install_locked(
                    "unused",
                    [record, dict(record, name="torch")],
                    Path(temp),
                    self.protected,
                )
            download.assert_not_called()

    def test_valid_dependency_archive(self):
        import zipfile

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "numpy-1.0-py3-none-any.whl"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr(
                    "numpy-1.0.dist-info/METADATA", "Name: numpy\nVersion: 1.0\n"
                )
                archive.writestr("numpy/__init__.py", "")
                archive.writestr("numpy-1.0.data/purelib/numpy_extra/__init__.py", "")
            self.tool.verify_download(
                path,
                {"sha256": self.tool.sha256(path), "name": "numpy", "version": "1.0"},
            )

    def test_staging_inputs(self):
        files = {
            "python/setup.py": "",
            "python/pyproject_other.toml": '[tool.sglang]\nrust-extensions = ["multimodal"]',
            "python/sglang/srt/rust_extensions/torch_build.py": "",
            "rust/Cargo.toml": '[workspace]\nmembers = ["sglang-mm"]',
            "rust/Cargo.lock": "",
            "rust/rust-toolchain.toml": '[toolchain]\nchannel = "1.92"',
            "rust/sglang-mm/Cargo.toml": '[package.metadata.sglang]\npython-module = "sglang.srt.rust_extensions._multimodal"',
            "rust/sglang-mm/src/lib.rs": "",
            "rust/sglang-radix-tree/Cargo.toml": "",
            "rust/sglang-radix-tree/Cargo.lock": "",
            "scripts/release/get_version_tag.py": "",
        }
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative, content in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content)
            self.tool.validate_staging(root)
            for relative in files:
                with self.subTest(missing=relative):
                    path = root / relative
                    saved = path.with_name(path.name + ".saved")
                    path.rename(saved)
                    try:
                        with self.assertRaises(ValueError):
                            self.tool.validate_staging(root)
                    finally:
                        saved.rename(path)
            (root / "stale.so").write_bytes(b"stale")
            with self.assertRaises(ValueError):
                self.tool.validate_staging(root)

    def test_reject_dependency_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "numpy-1.0-py3-none-any.whl"
            path.write_bytes(b"bad")
            with self.assertRaises(ValueError):
                self.tool.verify_download(
                    path, {"sha256": "a" * 64, "name": "numpy", "version": "1.0"}
                )


if __name__ == "__main__":
    unittest.main()
