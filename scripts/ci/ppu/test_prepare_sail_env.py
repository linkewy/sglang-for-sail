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

    def vision_record(self):
        return {
            "name": "torchvision",
            "version": "0.28.0+cpu",
            "url": "https://download-r2.pytorch.org/whl/cpu/torchvision-0.28.0%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl",
            "filename": "torchvision-0.28.0+cpu-cp312-cp312-manylinux_2_28_x86_64.whl",
            "sha256": "b545d46f4d2f9d30381281cf22874bfe1d32a8a7b0ee8396fccde89f30c6a9d9",
        }

    def test_torchvision_plan_and_lock_require_cpu_binary(self):
        expected = self.vision_record()
        candidates = [expected]
        for field, value in (
            ("version", "0.28.0"),
            ("url", expected["url"].replace("download-r2.pytorch.org", "example.org")),
            ("sha256", "a" * 64),
        ):
            candidates.append(dict(expected, **{field: value}))
        for record in candidates:
            item = {
                "metadata": {"name": record["name"], "version": record["version"]},
                "download_info": {
                    "url": record["url"],
                    "archive_info": {"hashes": {"sha256": record["sha256"]}},
                },
            }
            with self.subTest(record=record):
                if record == expected:
                    self.assertEqual(
                        self.tool.validate_plan(
                            {"install": [item]}, self.protected, {}
                        ),
                        [expected],
                    )
                    self.tool.validate_locked([record], self.protected)
                else:
                    for validator in ("plan", "lock"):
                        with (
                            self.subTest(validator=validator),
                            self.assertRaises(ValueError),
                        ):
                            if validator == "plan":
                                self.tool.validate_plan(
                                    {"install": [item]}, self.protected, {}
                                )
                            else:
                                self.tool.validate_locked([record], self.protected)

    def test_torchvision_check_executes_native_cpu_nms(self):
        from types import SimpleNamespace
        from unittest.mock import Mock, patch

        self.assertTrue(hasattr(self.tool, "check_torchvision"))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            vision = SimpleNamespace(
                __file__=str(root / "lib/site-packages/torchvision/__init__.py"),
                ops=SimpleNamespace(nms=Mock()),
            )
            torch = SimpleNamespace(
                tensor=Mock(side_effect=["boxes", "scores"]), float32="float32"
            )
            with (
                patch.object(self.tool.sys, "prefix", str(root)),
                patch.object(self.tool.metadata, "version", return_value="0.28.0+cpu"),
                patch.object(
                    self.tool.importlib,
                    "import_module",
                    side_effect={"torch": torch, "torchvision": vision}.__getitem__,
                ),
            ):
                for result in ([0, 2], [0, 1, 2]):
                    torch.tensor.reset_mock(side_effect=True)
                    torch.tensor.side_effect = ["boxes", "scores"]
                    vision.ops.nms.return_value.tolist.return_value = result
                    if result == [0, 2]:
                        self.tool.check_torchvision()
                    else:
                        with self.assertRaisesRegex(ValueError, "NMS"):
                            self.tool.check_torchvision()
                    vision.ops.nms.assert_called_with("boxes", "scores", 0.5)
                    self.assertEqual(
                        torch.tensor.call_args_list[0].args[0],
                        [[0, 0, 2, 2], [0, 0, 2, 2], [4, 4, 6, 6]],
                    )
                    self.assertEqual(
                        torch.tensor.call_args_list[1].args[0], [0.9, 0.8, 0.7]
                    )
                    for call in torch.tensor.call_args_list:
                        self.assertEqual(
                            call.kwargs, {"dtype": "float32", "device": "cpu"}
                        )
                vision.ops.nms.side_effect = RuntimeError("native operator unavailable")
                torch.tensor.side_effect = ["boxes", "scores"]
                with self.assertRaisesRegex(
                    RuntimeError, "native operator unavailable"
                ):
                    self.tool.check_torchvision()
                vision.ops.nms.reset_mock(side_effect=True)
                with (
                    patch.object(self.tool.metadata, "version", return_value="0.28.0"),
                    self.assertRaisesRegex(ValueError, "torchvision"),
                ):
                    self.tool.check_torchvision()
                vision.ops.nms.assert_not_called()
                vision.__file__ = "/system/site-packages/torchvision/__init__.py"
                with self.assertRaisesRegex(ValueError, "fresh venv"):
                    self.tool.check_torchvision()
                vision.ops.nms.assert_not_called()

    def test_test_entrypoints_import_without_running_device_tests(self):
        from unittest.mock import patch

        self.assertTrue(hasattr(self.tool, "check_test_entrypoints"))
        paths = ("test/run_suite.py", "test/registered/ppu/test_sail_qwen3_smoke.py")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in paths:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(
                    "from pathlib import Path\n"
                    "import unittest\n"
                    "Path(__file__).with_suffix('.imported').touch()\n"
                    "class DeviceTest(unittest.TestCase):\n"
                    "    @classmethod\n"
                    "    def setUpClass(cls):\n"
                    "        raise RuntimeError('device setup must not run')\n"
                    "    def test_device(self):\n"
                    "        raise RuntimeError('device test must not run')\n"
                    "if __name__ == '__main__':\n"
                    "    raise RuntimeError('entrypoint main must not run')\n"
                )
            with patch.object(self.tool, "SELF", root / "scripts/ci/ppu/prepare.py"):
                self.tool.check_test_entrypoints()
            for relative in paths:
                self.assertTrue((root / relative).with_suffix(".imported").is_file())

    def test_test_entrypoints_propagate_missing_dependencies(self):
        from unittest.mock import patch

        self.assertTrue(hasattr(self.tool, "check_test_entrypoints"))
        paths = ("test/run_suite.py", "test/registered/ppu/test_sail_qwen3_smoke.py")
        for failing in paths:
            with self.subTest(path=failing), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                for relative in paths:
                    path = root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(
                        "import sail_ci_deliberately_missing_dependency\n"
                        if relative == failing
                        else "pass\n"
                    )
                with (
                    patch.object(self.tool, "SELF", root / "scripts/ci/ppu/prepare.py"),
                    self.assertRaisesRegex(ModuleNotFoundError, "deliberately_missing"),
                ):
                    self.tool.check_test_entrypoints()

    def test_runtime_resolver_includes_image_protobuf_consumers(self):
        import ast

        tree = ast.parse(TOOL.read_text())
        runtime_call = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "resolve_install"
            and isinstance(node.args[1], ast.List)
            and any(
                isinstance(item, ast.Constant) and item.value == "sgl-eval==0.1.2"
                for item in node.args[1].elts
            )
        )
        requirements = {
            item.value
            for item in runtime_call.args[1].elts
            if isinstance(item, ast.Constant)
        }
        self.assertTrue(
            {
                "google-api-core>=2",
                "googleapis-common-protos",
                "google-api-python-client>=2",
            }
            <= requirements,
            "Resolve the image's protobuf consumers together with the runtime",
        )

    def cleanup_snapshot(self):
        return {
            "installed": {
                "cloud-tpu-client": {"version": "0.10", "path": "/sdk"},
                "torch": {"version": "2.13.0", "path": "/sdk"},
                "google-api-python-client": {"version": "1.8.0", "path": "/sdk"},
            },
            "requirements": {
                "cloud-tpu-client": ["google-api-python-client==1.8.0"],
                "torch": [],
                "google-api-python-client": [],
            },
            "protected": self.protected,
        }

    def test_image_cleanup_plan_is_narrow_and_fail_closed(self):
        import copy

        self.assertTrue(hasattr(self.tool, "image_cleanup_plan"))
        before = self.cleanup_snapshot()
        self.assertEqual(
            self.tool.image_cleanup_plan(before), {"cloud-tpu-client": "0.10"}
        )
        absent = copy.deepcopy(before)
        absent["installed"].pop("cloud-tpu-client")
        self.assertEqual(self.tool.image_cleanup_plan(absent), {})
        for requirement in (
            "cloud-tpu-client>=0.10",
            "Cloud_TPU_Client[extra] @ https://example.org/client.whl",
            'cloud-tpu-client; extra == "tpu"',
        ):
            bad = copy.deepcopy(before)
            bad["requirements"]["torch"] = [requirement]
            with (
                self.subTest(requirement=requirement),
                self.assertRaisesRegex(ValueError, "reverse dependency"),
            ):
                self.tool.image_cleanup_plan(bad)
        bad = copy.deepcopy(before)
        bad["installed"]["cloud-tpu-client"]["version"] = "0.11"
        with self.assertRaisesRegex(ValueError, "Unexpected cleanup version"):
            self.tool.image_cleanup_plan(bad)
        bad = copy.deepcopy(before)
        bad["protected"]["distributions"]["cloud-tpu-client"] = {"version": "0.10"}
        with self.assertRaisesRegex(ValueError, "protected"):
            self.tool.image_cleanup_plan(bad)

    def test_image_cleanup_requires_explicit_disposable_container(self):
        import os
        from unittest.mock import patch

        self.assertTrue(hasattr(self.tool, "normalize_ci_image"))
        before = self.cleanup_snapshot()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(self.tool, "run") as run,
        ):
            self.assertEqual(self.tool.normalize_ci_image(before), (before, {}))
            run.assert_not_called()
        for overrides in (
            {"SAIL_BUILD_IMAGE": "untrusted:latest"},
            {"KUBERNETES_SERVICE_HOST": ""},
        ):
            env = {
                "SAIL_ALLOW_IMAGE_CLEANUP": "1",
                "SAIL_BUILD_IMAGE": self.tool.CI_CLEANUP_IMAGE,
                "KUBERNETES_SERVICE_HOST": "test-cluster",
                **overrides,
            }
            with (
                self.subTest(overrides=overrides),
                patch.dict(os.environ, env, clear=True),
                patch.object(self.tool.sys, "platform", "linux"),
                patch.object(self.tool.sys, "prefix", self.tool.sys.base_prefix),
                patch.object(Path, "is_file", return_value=False),
                patch.object(self.tool, "run") as run,
                self.assertRaises(ValueError),
            ):
                self.tool.normalize_ci_image(before)
            run.assert_not_called()

    def test_image_cleanup_uninstalls_only_the_approved_package(self):
        import copy
        import os
        from unittest.mock import patch

        self.assertTrue(hasattr(self.tool, "normalize_ci_image"))
        before = self.cleanup_snapshot()
        after = copy.deepcopy(before)
        after["installed"].pop("cloud-tpu-client")
        after["requirements"].pop("cloud-tpu-client")
        env = {
            "SAIL_ALLOW_IMAGE_CLEANUP": "1",
            "SAIL_BUILD_IMAGE": self.tool.CI_CLEANUP_IMAGE,
            "KUBERNETES_SERVICE_HOST": "test-cluster",
        }
        with (
            patch.dict(os.environ, env, clear=True),
            patch.object(self.tool.sys, "platform", "linux"),
            patch.object(self.tool.sys, "prefix", self.tool.sys.base_prefix),
            patch.object(self.tool, "run") as run,
            patch.object(self.tool, "snapshot_using", return_value=after),
        ):
            actual, policy = self.tool.normalize_ci_image(before)
            self.assertEqual(actual, after)
            self.assertEqual(policy, {"cloud-tpu-client": "0.10"})
            run.assert_called_once_with(
                [
                    self.tool.sys.executable,
                    "-m",
                    "pip",
                    "uninstall",
                    "--yes",
                    "cloud-tpu-client",
                ]
            )
            run.reset_mock()
            self.assertEqual(self.tool.normalize_ci_image(after), (after, policy))
            run.assert_not_called()
            for changed in ("torch", "google-api-python-client"):
                bad = copy.deepcopy(after)
                bad["installed"][changed]["version"] = "changed"
                with (
                    self.subTest(changed=changed),
                    patch.object(self.tool, "snapshot_using", return_value=bad),
                    self.assertRaises(ValueError),
                ):
                    self.tool.normalize_ci_image(before)

    def test_cleanup_policy_must_match_replay(self):
        self.assertTrue(hasattr(self.tool, "validate_image_exclusions"))
        policy = {"cloud-tpu-client": "0.10"}
        self.tool.validate_image_exclusions(policy, policy, {"torch": {}})
        for expected, actual, installed in (
            (policy, {}, {}),
            ({}, policy, {}),
            (policy, policy, {"cloud-tpu-client": {"version": "0.10"}}),
        ):
            with (
                self.subTest(actual=actual, installed=installed),
                self.assertRaises(ValueError),
            ):
                self.tool.validate_image_exclusions(expected, actual, installed)

    def test_snapshot_collects_all_declared_reverse_dependencies(self):
        from types import SimpleNamespace
        from unittest.mock import patch

        def dist(name, requirements):
            return SimpleNamespace(
                metadata={"Name": name},
                version="1.0",
                requires=requirements,
                files=[f"{name}/__init__.py"],
                locate_file=lambda p: Path("/sdk") / p,
            )

        torch, triton = dist("torch", []), dist("triton", [])
        consumer = dist("consumer", ['cloud-tpu-client; extra == "tpu"'])
        duplicate = dist("consumer", ["Cloud_TPU_Client>=0.10"])
        with (
            patch.object(
                self.tool.metadata,
                "distributions",
                return_value=[torch, triton, consumer, duplicate],
            ),
            patch.object(
                self.tool.metadata,
                "packages_distributions",
                return_value={"torch": ["torch"], "triton": ["triton"]},
            ),
            patch.object(
                self.tool.metadata,
                "distribution",
                side_effect={"torch": torch, "triton": triton}.get,
            ),
            patch.object(
                self.tool.importlib,
                "import_module",
                side_effect=lambda n: SimpleNamespace(__file__=f"/sdk/{n}/__init__.py"),
            ),
        ):
            actual = self.tool.snapshot()
        self.assertEqual(
            actual.get("requirements", {}).get("consumer"),
            consumer.requires + duplicate.requires,
        )

    def test_install_only_never_resolves_or_normalizes_runtime(self):
        self.assertIn('"install-only"', TOOL.read_text())
        from unittest.mock import patch
        from test_sail_wheel_manifest import wheel

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifact = root / "artifact"
            artifact.mkdir()
            wheel(artifact, "sglang")
            wheel(artifact, "sglang_kernel")
            with (
                patch.object(
                    self.tool, "snapshot_using", return_value=self.cleanup_snapshot()
                ),
                patch.object(self.tool, "run") as run,
                patch.object(self.tool, "resolve_install") as resolve,
                patch.object(self.tool, "install_locked") as replay,
                patch.object(self.tool, "normalize_ci_image", create=True) as normalize,
            ):
                self.tool.prepare("install-only", root / "venv", artifact)
            resolve.assert_not_called()
            replay.assert_not_called()
            normalize.assert_not_called()
            commands = [[str(x) for x in call.args[0]] for call in run.call_args_list]
            install = next(cmd for cmd in commands if "install" in cmd)
            self.assertIn("--no-deps", install)
            self.assertIn("--no-index", install)
            self.assertFalse(any(cmd[-2:] == ["pip", "check"] for cmd in commands))
            check = next(cmd for cmd in commands if "check-installed" in cmd)
            self.assertIn("--static-only", check)
            self.assertFalse((artifact / "runtime-deps.lock.json").exists())

    def test_static_install_check_does_not_import_runtime(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import inspect

        self.assertIn(
            "static_only", inspect.signature(self.tool.check_installed).parameters
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            site = root / "lib/site-packages"
            site.mkdir(parents=True)
            native = site / "native.so"
            native.write_bytes(b"native")
            records = [
                {
                    "name": name,
                    "version": "1.0",
                    "native_files": {"native.so": self.tool.sha256(native)},
                }
                for name in ("sglang", "sglang-kernel")
            ]
            for name in ("sglang", "sglang_kernel"):
                (root / f"{name}.whl").touch()
            with (
                patch.object(self.tool.sys, "prefix", str(root)),
                patch.object(self.tool, "inspect_wheel", side_effect=records * 4),
                patch.object(
                    self.tool.metadata,
                    "distribution",
                    return_value=SimpleNamespace(
                        version="1.0", locate_file=lambda p: site / p
                    ),
                ),
                patch.object(
                    self.tool.importlib.util,
                    "find_spec",
                    return_value=SimpleNamespace(origin=str(site / "__init__.py")),
                ),
                patch.object(
                    self.tool, "check_torchvision", create=True
                ) as check_vision,
                patch.object(
                    self.tool, "check_test_entrypoints", create=True
                ) as check_entrypoints,
                patch.object(
                    self.tool.importlib,
                    "import_module",
                    return_value=SimpleNamespace(__file__=str(site / "__init__.py")),
                ) as import_runtime,
            ):
                self.tool.check_installed(root, static_only=True)
                check_vision.assert_not_called()
                check_entrypoints.assert_not_called()
                import_runtime.assert_not_called()
                self.tool.check_installed(root)
                check_vision.assert_called_once_with()
                import_runtime.assert_called_once_with("sglang")
                check_entrypoints.assert_called_once_with()
                check_entrypoints.side_effect = ModuleNotFoundError("tabulate")
                with self.assertRaisesRegex(ModuleNotFoundError, "tabulate"):
                    self.tool.check_installed(root)
                import_runtime.reset_mock()
                check_entrypoints.reset_mock(side_effect=True)
                check_vision.side_effect = RuntimeError("native operator unavailable")
                with self.assertRaisesRegex(
                    RuntimeError, "native operator unavailable"
                ):
                    self.tool.check_installed(root)
                import_runtime.assert_not_called()

    def test_replay_preflights_identity_policy_and_lock_before_cleanup(self):
        import json
        import os
        from unittest.mock import patch
        from test_sail_wheel_manifest import wheel
        import sail_wheel_manifest as manifest

        self.assertTrue(hasattr(self.tool, "normalize_ci_image"))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            wheel(root, "sglang")
            wheel(root, "sglang_kernel")
            before = self.cleanup_snapshot()
            identity = dict(
                git_sha="a" * 40,
                build_image=self.tool.CI_CLEANUP_IMAGE,
                sailify_revision=manifest.SAILIFY_REVISION,
                rust_toolchain="1.92",
                vendor_runtime={
                    "torch": {"distribution": "torch", "version": "2.13.0"},
                    "triton": {"distribution": "triton", "version": "3.7.0"},
                },
            )
            base_lock = dict(
                schema_version=1,
                protected=before["protected"],
                inherited={},
                wheels=[],
                image_exclusions={"cloud-tpu-client": "0.10"},
            )
            for kind in ("policy", "provider", "inherited", "locked", "identity"):
                lock = dict(base_lock)
                if kind == "policy":
                    lock["image_exclusions"] = {}
                elif kind == "provider":
                    lock["protected"] = {}
                elif kind == "inherited":
                    lock["inherited"] = {"missing": "1.0"}
                elif kind == "locked":
                    lock["wheels"] = [{"name": "torch"}]
                (root / "runtime-deps.lock.json").write_text(json.dumps(lock))
                manifest.create(root, identity)
                env = dict(
                    SAIL_ALLOW_IMAGE_CLEANUP="1",
                    SAIL_BUILD_IMAGE=self.tool.CI_CLEANUP_IMAGE,
                    SAIL_TESTED_SHA=("b" if kind == "identity" else "a") * 40,
                    KUBERNETES_SERVICE_HOST="test-cluster",
                )
                with (
                    self.subTest(kind=kind),
                    patch.dict(os.environ, env, clear=True),
                    patch.object(self.tool.sys, "platform", "linux"),
                    patch.object(self.tool.sys, "prefix", self.tool.sys.base_prefix),
                    patch.object(self.tool, "snapshot_using", return_value=before),
                    patch.object(self.tool, "normalize_ci_image") as normalize,
                    patch.object(self.tool, "run") as run,
                    self.assertRaises(ValueError),
                ):
                    self.tool.prepare("replay", root / "venv", root)
                normalize.assert_not_called()
                run.assert_not_called()

    def test_replay_checks_bootstrap_and_actual_venv_before_install(self):
        import copy
        import ensurepip
        import json
        import os
        from unittest.mock import patch
        import sail_wheel_manifest as manifest
        from test_sail_wheel_manifest import wheel

        for kind in ("bootstrap", "pip", "google-api-python-client"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                wheel(root, "sglang")
                wheel(root, "sglang_kernel")
                base = self.cleanup_snapshot()
                base["installed"].pop("cloud-tpu-client")
                base["requirements"].pop("cloud-tpu-client")
                base["installed"]["pip"] = {"version": "system-pip", "path": "/sdk"}
                current = copy.deepcopy(base)
                current["installed"]["pip"]["version"] = ensurepip.version()
                inherited = {
                    "pip": ensurepip.version(),
                    "google-api-python-client": "1.8.0",
                }
                if kind == "bootstrap":
                    inherited["pip"] = "wrong-bootstrap-version"
                else:
                    current["installed"][kind]["version"] = "changed-in-venv"
                lock = dict(
                    schema_version=1,
                    protected=base["protected"],
                    inherited=inherited,
                    wheels=[],
                    image_exclusions={},
                )
                (root / "runtime-deps.lock.json").write_text(json.dumps(lock))
                identity = dict(
                    git_sha="a" * 40,
                    build_image=self.tool.CI_CLEANUP_IMAGE,
                    sailify_revision=manifest.SAILIFY_REVISION,
                    rust_toolchain="1.92",
                    vendor_runtime={
                        "torch": {"distribution": "torch", "version": "2.13.0"},
                        "triton": {"distribution": "triton", "version": "3.7.0"},
                    },
                )
                manifest.create(root, identity)
                with (
                    patch.dict(
                        os.environ,
                        {
                            "SAIL_BUILD_IMAGE": identity["build_image"],
                            "SAIL_TESTED_SHA": identity["git_sha"],
                        },
                        clear=True,
                    ),
                    patch.object(
                        self.tool,
                        "snapshot_using",
                        side_effect=[base, current, current],
                    ),
                    patch.object(
                        self.tool, "normalize_ci_image", return_value=(base, {})
                    ) as normalize,
                    patch.object(self.tool, "install_locked") as install,
                    patch.object(self.tool, "run") as run,
                    self.assertRaisesRegex(
                        ValueError, "Inherited dependency differs"
                    ) as error,
                ):
                    self.tool.prepare("replay", root / "venv", root)
                self.assertIn(
                    "pip" if kind == "bootstrap" else kind, str(error.exception)
                )
                install.assert_not_called()
                if kind == "bootstrap":
                    normalize.assert_not_called()
                    run.assert_not_called()
                else:
                    normalize.assert_called_once()
                    self.assertEqual(len(run.call_args_list), 1)
                    self.assertIn("venv", run.call_args.args[0])

    def test_resolver_rejects_excluded_package_before_download(self):
        import inspect
        import json
        from unittest.mock import patch

        self.assertIn(
            "exclusions", inspect.signature(self.tool.resolve_install).parameters
        )
        item = self.entry(
            "cloud-tpu-client",
            "https://example.org/cloud_tpu_client-1.0-py3-none-any.whl",
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "pip-report.json").write_text(json.dumps({"install": [item]}))
            with (
                patch.object(self.tool, "run"),
                patch.object(self.tool, "install_locked") as install,
                self.assertRaisesRegex(ValueError, "Excluded package"),
            ):
                self.tool.resolve_install(
                    "unused",
                    [],
                    self.cleanup_snapshot(),
                    {},
                    root,
                    exclusions={"cloud-tpu-client": "0.10"},
                )
            install.assert_not_called()

    def test_runtime_promotion_and_replay_keep_wheel_identity(self):
        import copy
        import json
        import os
        import subprocess
        from unittest.mock import patch
        import sail_wheel_manifest as manifest
        from test_sail_wheel_manifest import wheel

        before = self.cleanup_snapshot()
        before["installed"]["triton"] = {"version": "3.7.0", "path": "/sdk"}
        before["requirements"]["triton"] = []
        before["protected"]["modules"]["triton"] = {
            "distribution": "triton",
            "version": "3.7.0",
            "path": "/sdk/triton/__init__.py",
        }
        after = copy.deepcopy(before)
        after["installed"].pop("cloud-tpu-client")
        after["requirements"].pop("cloud-tpu-client")
        import ensurepip

        for snapshot in (before, after):
            snapshot["installed"]["pip"] = {
                "version": "different-system-pip",
                "path": "/sdk",
            }
            snapshot["requirements"]["pip"] = []
        venv = copy.deepcopy(after)
        venv["installed"]["pip"] = {"version": ensurepip.version(), "path": "/venv"}
        policy = {"cloud-tpu-client": "0.10"}
        identity = dict(
            git_sha="a" * 40,
            build_image=self.tool.CI_CLEANUP_IMAGE,
            sailify_revision=manifest.SAILIFY_REVISION,
            rust_toolchain="1.92",
            vendor_runtime=before["protected"]["modules"],
        )
        for fail_check in ("pip", "entrypoints", None):
            with (
                self.subTest(fail_check=fail_check),
                tempfile.TemporaryDirectory() as temp,
            ):
                root = Path(temp)
                artifact = root / "artifact"
                artifact.mkdir()
                wheel(artifact, "sglang")
                wheel(artifact, "sglang_kernel")
                built = manifest.create(artifact, identity, stage="build")
                digest = manifest.sha256(artifact / "manifest.json")
                env = dict(
                    SAIL_ALLOW_IMAGE_CLEANUP="1",
                    SAIL_BUILD_IMAGE=self.tool.CI_CLEANUP_IMAGE,
                    SAIL_TESTED_SHA="a" * 40,
                    SAIL_VALIDATION_SHA="b" * 40,
                    SAIL_BUILD_RUN_ID="123",
                    SAIL_BUILD_MANIFEST_SHA256=digest,
                    KUBERNETES_SERVICE_HOST="test-cluster",
                )

                def execute(command, **kwargs):
                    if (
                        fail_check == "pip" and list(command)[-2:] == ["pip", "check"]
                    ) or (fail_check == "entrypoints" and "check-installed" in command):
                        raise subprocess.CalledProcessError(1, command)

                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.object(self.tool.sys, "platform", "linux"),
                    patch.object(self.tool.sys, "prefix", self.tool.sys.base_prefix),
                    patch.object(
                        self.tool, "snapshot_using", side_effect=[before, venv, venv]
                    ),
                    patch.object(
                        self.tool, "normalize_ci_image", return_value=(after, policy)
                    ) as normalize,
                    patch.object(
                        self.tool,
                        "resolve_install",
                        return_value=[self.vision_record()],
                    ) as resolve,
                    patch.object(self.tool, "run", side_effect=execute) as run,
                ):
                    if fail_check:
                        with self.assertRaises(subprocess.CalledProcessError):
                            self.tool.prepare("resolve", root / "resolve", artifact)
                    else:
                        self.tool.prepare("resolve", root / "resolve", artifact)
                normalize.assert_called_once_with(before)
                self.assertEqual(resolve.call_args.kwargs.get("exclusions"), policy)
                self.assertIn("tabulate", resolve.call_args.args[1])
                self.assertIn("pyarrow<26", resolve.call_args.args[1])
                vision = self.vision_record()
                self.assertIn(
                    f"torchvision @ {vision['url']}#sha256={vision['sha256']}",
                    resolve.call_args.args[1],
                )
                commands = [list(c.args[0]) for c in run.call_args_list]
                self.assertTrue(any(c[-2:] == ["pip", "check"] for c in commands))
                if fail_check:
                    self.assertFalse((artifact / "runtime-deps.lock.json").exists())
                    self.assertEqual(
                        manifest.sha256(artifact / "manifest.json"), digest
                    )
                    continue
                validated = manifest.verify(
                    artifact, identity["git_sha"], identity["build_image"]
                )
                self.assertEqual(validated["wheels"], built["wheels"])
                self.assertEqual(validated["build_manifest_sha256"], digest)
                self.assertEqual(validated.get("validation_sha"), "b" * 40)
                self.assertEqual(validated.get("build_run_id"), "123")
                self.assertEqual(validated["git_sha"], "a" * 40)
                lock = json.loads((artifact / "runtime-deps.lock.json").read_text())
                self.assertEqual(lock["image_exclusions"], policy)
                self.assertNotIn("cloud-tpu-client", lock["inherited"])
                with (
                    patch.dict(os.environ, env, clear=True),
                    patch.object(self.tool.sys, "platform", "linux"),
                    patch.object(self.tool.sys, "prefix", self.tool.sys.base_prefix),
                    patch.object(
                        self.tool, "snapshot_using", side_effect=[before, venv, venv]
                    ),
                    patch.object(
                        self.tool, "normalize_ci_image", return_value=(after, policy)
                    ),
                    patch.object(self.tool, "resolve_install") as resolve,
                    patch.object(self.tool, "install_locked") as install,
                    patch.object(self.tool, "run") as run,
                ):
                    try:
                        self.tool.prepare("replay", root / "replay", artifact)
                    except ValueError as exc:
                        self.fail(
                            f"Valid venv lock must replay despite system pip: {exc}"
                        )
                resolve.assert_not_called()
                install.assert_called_once()
                self.assertEqual(install.call_args.args[1], [self.vision_record()])
                self.assertEqual(lock["wheels"], [self.vision_record()])
                for validation_sha in ("c" * 40, "invalid"):
                    with (
                        self.subTest(validation_sha=validation_sha),
                        patch.dict(
                            os.environ,
                            dict(env, SAIL_VALIDATION_SHA=validation_sha),
                            clear=True,
                        ),
                        patch.object(self.tool, "snapshot_using", return_value=before),
                        patch.object(self.tool, "normalize_ci_image") as normalize,
                        patch.object(self.tool, "run") as rejected,
                        self.assertRaisesRegex(ValueError, "[Vv]alidation"),
                    ):
                        self.tool.prepare("replay", root / "rejected", artifact)
                    normalize.assert_not_called()
                    rejected.assert_not_called()
                self.assertTrue(
                    any(
                        list(c.args[0])[-2:] == ["pip", "check"]
                        for c in run.call_args_list
                    )
                )

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

    def test_downloader_identifies_client_and_checks_wheel(self):
        import hashlib
        import io
        import zipfile
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from urllib.request import Request

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "numpy-1.0.dist-info/METADATA", "Name: numpy\nVersion: 1.0\n"
            )
        payload = buffer.getvalue()
        record = {
            "name": "numpy",
            "version": "1.0",
            "url": "https://example.org/numpy-1.0-py3-none-any.whl",
            "filename": "numpy-1.0-py3-none-any.whl",
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

        def open_wheel(request, timeout):
            self.assertIsInstance(request, Request)
            self.assertEqual(request.full_url, record["url"])
            self.assertEqual(request.get_header("User-agent"), "sglang-sail-ci/1.0")
            self.assertEqual(timeout, 120)
            response = io.BytesIO(payload)
            response.url = record["url"]
            return response

        for valid in (True, False):
            with (
                self.subTest(valid_hash=valid),
                tempfile.TemporaryDirectory() as temp,
                patch.object(self.tool, "urlopen", side_effect=open_wheel),
                patch.object(self.tool, "run") as install,
                redirect_stdout(io.StringIO()) as output,
            ):
                candidate = dict(record, sha256=record["sha256"] if valid else "a" * 64)
                if valid:
                    self.tool.install_locked(
                        "python", [candidate], Path(temp), self.protected
                    )
                    install.assert_called_once_with(
                        [
                            "python",
                            "-m",
                            "pip",
                            "install",
                            "--no-index",
                            "--no-deps",
                            Path(temp) / record["filename"],
                        ]
                    )
                    self.assertIn(record["name"], output.getvalue())
                    self.assertIn(record["url"], output.getvalue())
                else:
                    with self.assertRaisesRegex(ValueError, "hash mismatch"):
                        self.tool.install_locked(
                            "python", [candidate], Path(temp), self.protected
                        )
                    install.assert_not_called()

    def test_download_error_identifies_safe_dependency_without_installing(self):
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch
        from urllib.error import HTTPError

        record = self.vision_record()
        error = HTTPError(record["url"], 403, "Forbidden", {}, None)
        with (
            tempfile.TemporaryDirectory() as temp,
            patch.object(self.tool, "urlopen", side_effect=error) as download,
            patch.object(self.tool, "run") as install,
            redirect_stdout(io.StringIO()),
        ):
            with self.assertRaises(Exception) as caught:
                self.tool.install_locked("python", [record], Path(temp), self.protected)
            self.assertIsInstance(caught.exception, RuntimeError)
            self.assertRegex(str(caught.exception), "torchvision.*HTTP 403")
            self.assertIn(record["url"], str(caught.exception))
            download.assert_called_once()
            install.assert_not_called()
            self.assertEqual(list(Path(temp).iterdir()), [])

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
