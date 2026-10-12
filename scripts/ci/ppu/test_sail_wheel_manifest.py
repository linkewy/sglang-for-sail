import copy
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

TOOL = Path(__file__).with_name("sail_wheel_manifest.py")


def wheel(
    root, name, tag="cp312-cp312-linux_x86_64", version="1.0", metadata_name=None
):
    path = root / f"{name}-{version}-{tag}.whl"
    info = f"{name}-{version}.dist-info"
    extension = (
        "sgl_kernel/common_ops.cpython-312-x86_64-linux-gnu.so"
        if name == "sglang_kernel"
        else "sglang/srt/rust_extensions/_multimodal.cpython-312-x86_64-linux-gnu.so"
    )
    elf = b"\x7fELF\x02\x01" + bytes(12) + b"\x3e\x00" + bytes(44)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            f"{info}/METADATA", f"Name: {metadata_name or name}\nVersion: {version}\n"
        )
        archive.writestr(f"{info}/WHEEL", f"Wheel-Version: 1.0\nTag: {tag}\n")
        archive.writestr(extension, elf)
    return path


class TestManifest(unittest.TestCase):
    def setUp(self):
        self.assertTrue(TOOL.is_file(), "manifest tool is missing")
        spec = importlib.util.spec_from_file_location("manifest", TOOL)
        self.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tool)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        wheel(self.root, "sglang")
        wheel(self.root, "sglang_kernel")
        (self.root / "runtime-deps.lock.json").write_text(
            json.dumps({"schema_version": 1})
        )
        self.sha = "a" * 40
        self.image = "registry/image@sha256:" + "b" * 64
        self.identity = {
            "git_sha": self.sha,
            "build_image": self.image,
            "sailify_revision": "f401ecae76d7b4991d44920bde9edaac7e655fd8",
            "rust_toolchain": "1.92",
            "vendor_runtime": {
                "torch": {"distribution": "torch", "version": "2.13.0"},
                "triton": {"distribution": "triton", "version": "3.7.0"},
            },
        }

    def create(self):
        return self.tool.create(self.root, self.identity)

    def verify(self):
        return self.tool.verify(self.root, self.sha, self.image)

    def test_round_trip(self):
        self.create()
        document = self.verify()
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(len(document["wheels"]), 2)
        self.assertTrue(all(item["native_files"] for item in document["wheels"]))

    def test_tampering_and_missing_files(self):
        self.create()
        for name in [
            "runtime-deps.lock.json",
            "sglang-1.0-cp312-cp312-linux_x86_64.whl",
        ]:
            path = self.root / name
            original = path.read_bytes()
            path.write_bytes(b"tampered")
            with self.assertRaises(ValueError):
                self.verify()
            path.write_bytes(original)
        (self.root / "runtime-deps.lock.json").rename(self.root / "missing.lock")
        with self.assertRaises(ValueError):
            self.verify()

    def test_independent_identity(self):
        self.create()
        for sha, image in [("c" * 40, self.image), (self.sha, self.image[:-1] + "c")]:
            with self.assertRaises(ValueError):
                self.tool.verify(self.root, sha, image)

    def test_metadata_and_abi(self):
        for tag, metadata in [
            ("cp310-abi3-linux_x86_64", None),
            ("cp312-cp312-linux_aarch64", None),
            ("cp312-cp312-linux_x86_64", "other"),
        ]:
            with (
                self.subTest(tag=tag, metadata=metadata),
                tempfile.TemporaryDirectory() as temp,
            ):
                path = wheel(Path(temp), "sglang_kernel", tag, metadata_name=metadata)
                with self.assertRaises(ValueError):
                    self.tool.inspect_wheel(path)

    def test_extra_wheel(self):
        wheel(self.root, "sglang", version="2.0")
        with self.assertRaises(ValueError):
            self.create()

    def test_manifest_schema_paths_and_duplicates(self):
        doc = self.create()
        for mutate in (
            lambda d: d.update(schema_version=2),
            lambda d: d["wheels"][0].update(filename="../evil.whl"),
            lambda d: d["wheels"][0].update(filename="/tmp/evil.whl"),
            lambda d: d["wheels"].append(copy.deepcopy(d["wheels"][0])),
        ):
            bad = copy.deepcopy(doc)
            mutate(bad)
            (self.root / "manifest.json").write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                self.verify()

    def test_symlink_and_zip_traversal(self):
        path = self.root / "sglang-1.0-cp312-cp312-linux_x86_64.whl"
        with zipfile.ZipFile(path, "a") as archive:
            archive.writestr("../outside", "bad")
        with self.assertRaises(ValueError):
            self.tool.inspect_wheel(path)
        link = self.root / "alias.whl"
        link.symlink_to(path)
        with self.assertRaises(ValueError):
            self.tool.inspect_wheel(link)


if __name__ == "__main__":
    unittest.main()
