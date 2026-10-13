import ast
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts/ci/ppu"


class TestBuildContract(unittest.TestCase):
    def test_build_inputs_and_isolation(self):
        path = TOOLS / "build_sail_wheels.sh"
        self.assertTrue(path.is_file(), "build script is missing")
        source = path.read_text()
        for required in (
            "set -euo pipefail",
            'archive "$SAIL_TESTED_SHA"',
            "check-staging",
            'PYTORCH_SAIL_ARCH="ppu_10;ppu_15"',
            "SGL_KERNEL_COMPILE_THREADS",
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG",
            "--no-isolation",
            "setup_sail.py bdist_wheel",
            "pyproject_other.toml",
            "pyproject_sail.toml",
            "f401ecae76d7b4991d44920bde9edaac7e655fd8",
            "install-only",
            "--stage build",
            "rustup toolchain install 1.92",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)
        self.assertNotIn("SGLANG_BUILD_RUST_EXTS=none", source)
        result = subprocess.run(["bash", str(path)], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Usage:", result.stderr)

    def test_cpu_checks_collect_without_runtime_import(self):
        import sys

        for relative in (
            "test/registered/unit/test_sail_dispatch.py",
            "test/registered/unit/multimodal/test_internvl_utils_import.py",
        ):
            with self.subTest(path=relative):
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        "import runpy,sys; sys.modules['sglang'] = None; runpy.run_path(sys.argv[1])",
                        str(ROOT / relative),
                    ],
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_shell_syntax_checks_each_script(self):
        import shlex

        import yaml

        workflow = yaml.safe_load(
            (ROOT / ".github/workflows/_pr-test-ppu-build.yml").read_text()
        )
        commands = "\n".join(
            s.get("run", "") for s in workflow["jobs"]["lint"]["steps"]
        )
        checked = []
        for line in commands.splitlines():
            if not line.strip().startswith("bash -n "):
                continue
            words = shlex.split(line)
            if words[:2] == ["bash", "-n"]:
                self.assertEqual(len(words), 3, "bash -n only parses its first script")
                checked.append(words[2])
        self.assertEqual(
            set(checked),
            {"scripts/ci/ppu/build_sail_wheels.sh", "scripts/ci/ppu/run_sail_smoke.sh"},
        )

    def test_compile_thread_limit(self):
        import os

        source = (TOOLS / "build_sail_wheels.sh").read_text()
        begin = source.index("export SGL_KERNEL_COMPILE_THREADS=")
        end = source.index("export MAX_JOBS=")
        script = source[begin:end] + "\nexit 0\n"
        for value in ("0", "17", "invalid", "9999999999999999999999999999999"):
            result = subprocess.run(
                ["bash", "-e", "-c", script],
                capture_output=True,
                env=dict(os.environ, SGL_KERNEL_COMPILE_THREADS=value),
            )
            with self.subTest(value=value):
                self.assertNotEqual(result.returncode, 0)

    def test_build_rejects_non_linux(self):
        import sys

        source = (
            (TOOLS / "build_sail_wheels.sh")
            .read_text()
            .split("python3 - <<'PY'\n", 1)[1]
            .split("\nPY", 1)[0]
        )
        setup = (
            "import sys,types,platform; sys.platform='darwin'; platform.machine=lambda:'x86_64'; "
            "m=types.ModuleType('torch.utils.cpp_extension'); "
            "m._get_cuda_arch_flags=lambda:['--gpu-architecture=ppu_10,ppu_15']; "
            "sys.modules['torch.utils.cpp_extension']=m; "
        )
        result = subprocess.run(
            [sys.executable, "-c", setup + "\n" + source],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("linux/x86_64", result.stderr)

    def test_kernel_abi(self):
        source = (ROOT / "python/sglang/kernels/aot/setup_sail.py").read_text()
        ast.parse(source)
        self.assertNotIn('"py_limited_api": "cp310"', source)
        self.assertIn("py_limited_api=False", source)


class TestSmokeChecks(unittest.TestCase):
    def setUp(self):
        import importlib.util

        path = TOOLS / "sail_smoke_checks.py"
        self.assertTrue(path.is_file(), "smoke checks are missing")
        spec = importlib.util.spec_from_file_location("smoke_checks", path)
        self.tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.tool)
        self.identity = dict(
            run_token="new-run",
            git_sha="a" * 40,
            manifest_sha256="b" * 64,
            expected_device="ZW810E",
        )
        self.native = "sgl_kernel/common_ops.cpython-312-x86_64-linux-gnu.so"
        self.manifest = {
            "wheels": [
                {"name": "sglang-kernel", "native_files": {self.native: "c" * 64}}
            ]
        }
        self.result = dict(
            self.identity,
            schema_version=1,
            tests_run=2,
            failures=0,
            errors=0,
            skipped=0,
            successful_tests=["test_generate", "test_kernel_ops"],
            device="ZW810E",
            http={"status": 200, "text_nonempty": True, "completion_tokens": 16},
            native={"relative_path": self.native, "sha256": "c" * 64},
            cleanup_ok=True,
        )

    def test_result_rejects_empty_skip_old_or_partial(self):
        self.tool.validate_result(self.result, self.identity, self.manifest)
        for key, value in [
            ("tests_run", 0),
            ("skipped", 2),
            ("run_token", "old"),
            ("manifest_sha256", "x" * 64),
            ("git_sha", "d" * 40),
            ("successful_tests", ["test_generate"]),
            ("device", "NVIDIA H100"),
            ("cleanup_ok", False),
            ("errors", 1),
            ("failures", 1),
        ]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.tool.validate_result(
                    dict(self.result, **{key: value}), self.identity, self.manifest
                )
        for field in ("http", "native"):
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.tool.validate_result(
                    dict(self.result, **{field: {}}), self.identity, self.manifest
                )

    def test_registration(self):
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location(
            "ppu_registry_test", ROOT / "python/sglang/test/ci/ci_register.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        path = ROOT / "test/registered/ppu/test_sail_qwen3_smoke.py"
        self.assertTrue(path.is_file(), "registered smoke is missing")
        entries, main = module.ut_parse_one_file(str(path))
        self.tool.validate_registration(entries, main)
        self.assertEqual(entries[0].est_time, 300)
        for entries, main in [([], True), (entries, False)]:
            with self.assertRaises(ValueError):
                self.tool.validate_registration(entries, main)

    def test_smoke_launch_contract(self):
        tree = ast.parse(
            (ROOT / "test/registered/ppu/test_sail_qwen3_smoke.py").read_text()
        )
        launch = next(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "popen_launch_server"
        )
        args = ast.literal_eval(
            next(k.value for k in launch.keywords if k.arg == "other_args")
        )
        self.assertNotIn("--disable-piecewise-cuda-graph", args)
        for flag, value in (
            ("--attention-backend", "triton"),
            ("--sampling-backend", "pytorch"),
            ("--cuda-graph-backend-prefill", "disabled"),
            ("--cuda-graph-backend-decode", "disabled"),
        ):
            self.assertIn(flag, args)
            self.assertEqual(args[args.index(flag) + 1], value)
        self.assertEqual(
            ast.literal_eval(
                next(k.value for k in launch.keywords if k.arg == "timeout")
            ),
            600,
        )
        self.assertEqual(
            ast.literal_eval(next(k.value for k in launch.keywords if k.arg == "env")),
            {"HF_HUB_OFFLINE": "0", "SGLANG_TEST_MAX_RETRY": "0"},
        )
        case = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "TestSailQwen3Smoke"
        )
        self.assertEqual(ast.unparse(case.bases[0]), "unittest.TestCase")
        self.assertEqual(
            {
                n.name
                for n in case.body
                if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
            },
            {"test_generate", "test_kernel_ops"},
        )

    def test_pod_identity(self):
        import copy

        image = "example.org/sdk@sha256:" + "a" * 64
        pod = {
            "metadata": {
                "name": "job-worker-0",
                "uid": "uid",
                "namespace": "default",
                "labels": {"ppu-job": "job"},
            },
            "spec": {
                "nodeSelector": {"board-type": "OAM-810E"},
                "containers": [
                    {
                        "name": "worker",
                        "image": image,
                        "env": [{"name": "SAIL_SMOKE_RUN_TOKEN", "value": "new-run"}],
                    }
                ],
            },
            "status": {"containerStatuses": [{"name": "worker", "imageID": image}]},
        }
        evidence = self.tool.pod_evidence(
            [pod], "new-run", image, "board-type=OAM-810E"
        )
        self.assertEqual(evidence["pod"], "job-worker-0")
        self.assertIsNone(
            self.tool.pod_evidence([pod], "other-run", image, "board-type=OAM-810E")
        )
        for field in ("imageID",):
            bad = copy.deepcopy(pod)
            bad["status"]["containerStatuses"][0][field] = "wrong@sha256:" + "b" * 64
            with self.assertRaises(ValueError):
                self.tool.pod_evidence([bad], "new-run", image, "board-type=OAM-810E")
        with self.assertRaises(ValueError):
            self.tool.pod_evidence([pod, pod], "new-run", image, "board-type=OAM-810E")
        with self.assertRaises(ValueError):
            self.tool.pod_evidence([pod], "new-run", image, "board-type=ZW-M890P")


class TestWorkflowContract(unittest.TestCase):
    def load(self, name):
        import yaml

        path = ROOT / ".github/workflows" / name
        self.assertTrue(path.is_file(), f"{name} is missing")
        return yaml.safe_load(path.read_text())

    def test_build_and_runtime_jobs_are_separate(self):
        workflow = self.load("_pr-test-ppu-build.yml")
        jobs = workflow["jobs"]
        self.assertIn("runtime", jobs)
        build, runtime = jobs["build"], jobs["runtime"]
        self.assertEqual(runtime["needs"], "build")
        self.assertEqual(runtime["runs-on"], "ubuntu-24.04")
        self.assertEqual(runtime["container"]["image"], "${{ inputs.build_image }}")
        self.assertNotIn("SAIL_ALLOW_IMAGE_CLEANUP", build["env"])
        self.assertEqual(runtime["env"]["SAIL_ALLOW_IMAGE_CLEANUP"], "1")
        self.assertEqual(
            runtime["env"]["SAIL_BUILD_MANIFEST_SHA256"],
            "${{ needs.build.outputs.manifest_sha256 }}",
        )
        self.assertEqual(
            workflow[True]["workflow_call"]["outputs"]["manifest_sha256"]["value"],
            "${{ jobs.runtime.outputs.manifest_sha256 }}",
        )
        commands = "\n".join(s.get("run", "") for s in runtime["steps"])
        self.assertIn("prepare_sail_env.py resolve", commands)
        self.assertNotIn("build_sail_wheels.sh", commands)
        self.assertNotIn(
            '"$helper" resolve', (TOOLS / "build_sail_wheels.sh").read_text()
        )
        raw = next(
            s for s in build["steps"] if s.get("uses") == "actions/upload-artifact@v4"
        )
        download = next(
            s
            for s in runtime["steps"]
            if s.get("uses") == "actions/download-artifact@v4"
        )
        self.assertEqual(raw["with"]["name"], "wheel-python3.12-sail-build")
        self.assertEqual(raw["with"]["name"], download["with"]["name"])
        self.assertEqual(raw["with"]["if-no-files-found"], "error")
        smoke = self.load("_pr-test-ppu-smoke.yml")["jobs"]["smoke"]
        self.assertEqual(smoke["env"]["SAIL_ALLOW_IMAGE_CLEANUP"], "1")
        scheduler = next(s for s in smoke["steps"] if s.get("id") == "scheduler")
        self.assertIn("SAIL_ALLOW_IMAGE_CLEANUP=1", scheduler["with"]["extra_env"])

    def test_manual_reuse_skips_compilation_and_pins_source(self):
        main = self.load("pr-test-ppu.yml")
        inputs = main[True]["workflow_dispatch"]["inputs"]
        for key in ("reuse_build_run_id", "reuse_build_manifest_sha256"):
            self.assertIn(key, inputs)
            self.assertEqual(inputs[key]["default"], "")
            self.assertNotIn(key, main[True]["workflow_call"]["inputs"])
        build = self.load("_pr-test-ppu-build.yml")
        self.assertEqual(build["permissions"]["actions"], "read")
        steps = build["jobs"]["build"]["steps"]
        for step in steps:
            if "build_sail_wheels.sh" in step.get(
                "run", ""
            ) or "rust-toolchain@" in step.get("uses", ""):
                self.assertEqual(step.get("if"), "inputs.reuse_build_run_id == ''")
        download = next(s for s in steps if s.get("id") == "reuse_download")
        self.assertEqual(download["if"], "inputs.reuse_build_run_id != ''")
        self.assertEqual(download["with"]["run-id"], "${{ inputs.reuse_build_run_id }}")
        self.assertEqual(download["with"].get("name"), "wheel-python3.12-sail-build")
        self.assertNotIn("artifact-ids", download["with"])
        self.assertEqual(download["with"]["github-token"], "${{ github.token }}")
        manifest = next(s for s in steps if s.get("id") == "manifest")
        self.assertIn('"$REUSE_BUILD_MANIFEST_SHA256"', manifest["run"])
        for name in ("ppu-wheel-build", "ppu-smoke"):
            self.assertEqual(
                main["jobs"][name]["with"]["validation_sha"],
                "${{ needs.check-changes.outputs.validation_sha }}",
            )
        smoke = self.load("_pr-test-ppu-smoke.yml")["jobs"]["smoke"]
        self.assertEqual(
            smoke["env"]["SAIL_VALIDATION_SHA"],
            "${{ inputs.validation_sha || inputs.tested_sha }}",
        )
        self.assertIn(
            "SAIL_VALIDATION_SHA=",
            next(s for s in smoke["steps"] if s.get("id") == "scheduler")["with"][
                "extra_env"
            ],
        )
        self.assertIn(
            "${SAIL_VALIDATION_SHA:-$SAIL_TESTED_SHA}",
            (TOOLS / "run_sail_smoke.sh").read_text(),
        )

    def test_reuse_source_rejects_wrong_run_artifact_or_missing_compilation(self):
        import copy
        import json
        import os

        steps = self.load("_pr-test-ppu-build.yml")["jobs"]["build"]["steps"]
        self.assertTrue(
            any(s.get("id") == "reuse" for s in steps), "reuse preflight is missing"
        )
        script = next(s for s in steps if s.get("id") == "reuse")["with"]["script"]
        import shutil

        node = shutil.which("node") or next(
            (
                str(p)
                for p in (Path("/__e/node24/bin/node"), Path("/__e/node20/bin/node"))
                if p.is_file()
            ),
            None,
        )
        self.assertIsNotNone(node, "GitHub Actions Node runtime is required")
        data = {
            "run": {
                "id": 123,
                "status": "completed",
                "head_sha": "a" * 40,
                "path": ".github/workflows/pr-test-ppu.yml",
                "repository": {"full_name": "owner/repo"},
                "head_repository": {"full_name": "owner/repo"},
            },
            "artifacts": [
                {"id": 456, "name": "wheel-python3.12-sail-build", "expired": False}
            ],
            "jobs": [
                {
                    "name": "ppu-wheel-build / build",
                    "conclusion": "success",
                    "steps": [
                        {
                            "name": "Compile and validate two wheels without runtime dependencies",
                            "conclusion": "success",
                        }
                    ],
                }
            ],
        }
        cases = [("valid", data)]
        for field, value in (
            ("head_sha", "b" * 40),
            ("status", "in_progress"),
            ("path", "other.yml"),
            ("head_repository", {"full_name": "other/repo"}),
        ):
            bad = copy.deepcopy(data)
            bad["run"][field] = value
            cases.append((field, bad))
        for field, value in (("expired", True), ("name", "wheel-python3.12-sail")):
            bad = copy.deepcopy(data)
            bad["artifacts"][0][field] = value
            cases.append((field, bad))
        for value in ("failure", "skipped"):
            bad = copy.deepcopy(data)
            bad["jobs"][0]["steps"][0]["conclusion"] = value
            cases.append((value, bad))
        for name, records in cases:
            env = dict(
                os.environ,
                REUSE_BUILD_RUN_ID="123",
                SAIL_TESTED_SHA="a" * 40,
                FIXTURE=json.dumps(records),
            )
            harness = """
const fixture = JSON.parse(process.env.FIXTURE);
const context = {repo: {owner: 'owner', repo: 'repo'}, eventName: 'workflow_dispatch'};
const core = {setOutput: (key, value) => console.log(JSON.stringify({[key]: value}))};
const github = {rest: {actions: {
  getWorkflowRun: async () => ({data: fixture.run}),
  listWorkflowRunArtifacts: async () => ({data: {artifacts: fixture.artifacts}}),
  listJobsForWorkflowRun: async () => ({data: {jobs: fixture.jobs}}),
}}};
(async () => {
""" + script + "\n})().catch(e => {console.error(e.message); process.exitCode = 1;});"
            with self.subTest(case=name):
                result = subprocess.run(
                    [node, "-e", harness], env=env, capture_output=True, text=True
                )
                if name == "valid":
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn('"artifact_id":456', result.stdout)
                else:
                    self.assertNotEqual(result.returncode, 0)

    def test_reuse_identity_rejects_source_changes_and_nonmanual_events(self):
        import os
        import tempfile

        steps = self.load("pr-test-ppu.yml")["jobs"]["check-changes"]["steps"]
        step = next(s for s in steps if s.get("id") == "identity")
        self.assertIn("REUSE_BUILD_RUN_ID", step.get("env", {}))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = dict(
                os.environ,
                GIT_AUTHOR_NAME="Test",
                GIT_AUTHOR_EMAIL="test@example.org",
                GIT_COMMITTER_NAME="Test",
                GIT_COMMITTER_EMAIL="test@example.org",
                GITHUB_OUTPUT=str(root / "output"),
                REUSE_BUILD_RUN_ID="123",
                REUSE_BUILD_MANIFEST_SHA256="c" * 64,
                EVENT_NAME="workflow_dispatch",
            )

            def git(*args):
                return subprocess.run(
                    ["git", *args],
                    cwd=root,
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True,
                ).stdout.strip()

            git("init", "-q")
            (root / "source.py").write_text("original\n")
            git("add", ".")
            git("commit", "-qm", "baseline")
            built = git("rev-parse", "HEAD")
            helper = root / "scripts/ci/ppu/prepare_sail_env.py"
            helper.parent.mkdir(parents=True)
            helper.write_text("fixed\n")
            git("add", "scripts")
            git("commit", "-qm", "download fix")
            validation = git("rev-parse", "HEAD")
            env["TESTED_REF"] = built

            def execute(settings):
                return subprocess.run(
                    ["bash", "-e", "-c", step["run"]],
                    cwd=root,
                    env=settings,
                    capture_output=True,
                    text=True,
                )

            result = execute(env)
            self.assertEqual(result.returncode, 0, result.stderr)
            output = (root / "output").read_text()
            self.assertIn(f"tested_sha={built}", output)
            self.assertIn(f"validation_sha={validation}", output)
            for override in (
                {"EVENT_NAME": "pull_request"},
                {"TESTED_REF": "main"},
                {"REUSE_BUILD_MANIFEST_SHA256": "bad"},
                {"REUSE_BUILD_RUN_ID": "-1"},
            ):
                with self.subTest(override=override):
                    self.assertNotEqual(execute(dict(env, **override)).returncode, 0)
            (root / "source.py").write_text("changed\n")
            git("add", "source.py")
            git("commit", "-qm", "source change")
            self.assertNotEqual(execute(env).returncode, 0)
            normal = dict(
                env,
                REUSE_BUILD_RUN_ID="",
                REUSE_BUILD_MANIFEST_SHA256="",
                EVENT_NAME="pull_request",
            )
            self.assertEqual(execute(normal).returncode, 0)

    def test_build_uses_explicit_public_dependency_index(self):
        build = self.load("_pr-test-ppu-build.yml")["jobs"]["build"]
        self.assertEqual(build["env"].get("PIP_INDEX_URL"), "https://pypi.org/simple")

    def test_build_trusts_only_the_mounted_checkout(self):
        import os
        import tempfile

        build = self.load("_pr-test-ppu-build.yml")["jobs"]["build"]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            checkout = root / "checkout"
            other = root / "other"
            env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
            env.update(
                HOME=str(root),
                GIT_CONFIG_NOSYSTEM="1",
                GIT_CONFIG_GLOBAL=str(root / "absent-global-config"),
            )
            for repo in (checkout, other):
                subprocess.run(["git", "init", "-q", str(repo)], env=env, check=True)
            env["GIT_TEST_ASSUME_DIFFERENT_OWNER"] = "1"

            def check(repo, settings):
                return subprocess.run(
                    ["git", "-C", str(repo), "rev-parse", "--show-toplevel"],
                    env=settings,
                    capture_output=True,
                    text=True,
                )

            self.assertIn("dubious ownership", check(checkout, env).stderr)
            scoped = dict(env)
            scoped.update(
                {
                    key: value.replace("${{ github.workspace }}", str(checkout))
                    for key, value in build["env"].items()
                    if key.startswith("GIT_CONFIG_")
                }
            )
            result = check(checkout, scoped)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), str(checkout))
            self.assertIn("dubious ownership", check(other, scoped).stderr)
            self.assertIn("dubious ownership", check(checkout, env).stderr)
            self.assertFalse((root / "absent-global-config").exists())

    def test_hosted_build_resource_contract(self):
        build = self.load("_pr-test-ppu-build.yml")["jobs"]["build"]
        self.assertEqual(build["runs-on"], "ubuntu-24.04")
        self.assertEqual(build["container"]["image"], "${{ inputs.build_image }}")
        self.assertEqual(build["env"].get("SGL_KERNEL_COMPILE_THREADS"), "2")
        self.assertEqual(build["env"].get("CARGO_BUILD_JOBS"), "2")
        diagnostics = next(s for s in build["steps"] if s.get("id") == "resources")
        for command in ("uname -m", "nproc", "free -h", "df -h"):
            self.assertIn(command, diagnostics["run"])
        self.assertFalse(
            any("ppu-scheduler-action" in s.get("uses", "") for s in build["steps"])
        )

    def test_workflow_topology_and_credentials(self):
        main = self.load("pr-test-ppu.yml")["jobs"]
        self.assertIn("ppu-wheel-build", main)
        self.assertEqual(main["ppu-wheel-build"]["needs"], ["check-changes", "pr-gate"])
        self.assertTrue(main["pr-gate"]["with"]["require-run-ci"])
        self.assertEqual(main["pr-gate"]["with"]["require-label"], "run-ci-ppu")
        self.assertNotIn("secrets", main["pr-gate"])
        smoke = main["ppu-smoke"]
        self.assertFalse(smoke["strategy"]["fail-fast"])
        self.assertEqual(
            {m["expected_device_name"] for m in smoke["strategy"]["matrix"]["include"]},
            {"ZW810E", "ZW-M890P"},
        )
        for name in ("_pr-test-ppu-build.yml", "_pr-test-ppu-smoke.yml"):
            jobs = self.load(name)["jobs"]
            for job_name, job in jobs.items():
                if job_name == "lint":
                    self.assertEqual(job["runs-on"], "ubuntu-latest")
                    self.assertEqual(jobs["build"]["needs"], "lint")
                else:
                    expected_runner = (
                        "ubuntu-24.04"
                        if job_name in ("build", "runtime")
                        else "k8s-runner-group-cpu-thead"
                    )
                    self.assertEqual(job["runs-on"], expected_runner)
                    self.assertIn("container", job)
                self.assertNotIn("continue-on-error", job)
                for step in job["steps"]:
                    if step.get("uses") == "actions/checkout@v4":
                        self.assertIs(step["with"]["persist-credentials"], False)
                        self.assertEqual(
                            step["with"]["ref"],
                            "${{ inputs.validation_sha || inputs.tested_sha }}",
                        )
        build = self.load("_pr-test-ppu-build.yml")["jobs"]["build"]
        self.assertEqual(build["timeout-minutes"], 120)
        self.assertTrue(
            any(
                s.get("uses")
                == "dtolnay/rust-toolchain@0be0c8e6607124ba7557736a4b6b5a8fe2b0c284"
                for s in build["steps"]
            ),
            "fixed Rust bootstrap is missing",
        )
        smoke = self.load("_pr-test-ppu-smoke.yml")["jobs"]["smoke"]
        self.assertEqual(smoke["timeout-minutes"], 90)
        action = next(s for s in smoke["steps"] if s.get("id") == "scheduler")
        self.assertEqual(
            action["uses"],
            "t-head/ppu-scheduler-action@main",
        )
        self.assertEqual(action["timeout-minutes"], 80)
        self.assertEqual(action["with"]["timeout_minutes"], 75)
        self.assertEqual(action["with"]["cleanup_policy"], "always")
        self.assertEqual(action["with"]["nnodes"], 0)
        self.assertEqual(action["with"]["nproc_per_node"], 1)

    def test_path_filter(self):
        import fnmatch

        import yaml

        steps = self.load("pr-test-ppu.yml")["jobs"]["check-changes"]["steps"]
        filters = yaml.safe_load(
            next(s for s in steps if s.get("id") == "filter")["with"]["filters"]
        )["ppu"]
        for path in (
            "rust/Cargo.toml",
            "rust/sglang-radix-tree/Cargo.lock",
            "python/pyproject_other.toml",
            "python/sglang/srt/models/qwen3.py",
            "python/sglang/test/test_utils.py",
            "scripts/release/get_version_tag.py",
            ".github/workflows/pr-gate.yml",
            "test/run_suite.py",
            "test/registered/unit/test_sail_dispatch.py",
        ):
            with self.subTest(path=path):
                self.assertTrue(any(fnmatch.fnmatchcase(path, p) for p in filters))
        self.assertFalse(
            any(fnmatch.fnmatchcase("docs/unrelated.md", p) for p in filters)
        )

    def test_finish_truth_table(self):
        import json
        import os

        jobs = self.load("pr-test-ppu.yml")["jobs"]
        step = next(
            s for s in jobs["pr-test-ppu-finish"]["steps"] if s.get("id") == "statuses"
        )
        base = {
            key: {"result": "success"}
            for key in ("check-changes", "pr-gate", "ppu-wheel-build", "ppu-smoke")
        }
        base["check-changes"]["outputs"] = {"ppu": "true"}

        def execute(needs):
            return subprocess.run(
                ["bash", "-e", "-c", step["run"]],
                capture_output=True,
                text=True,
                env=dict(os.environ, NEEDS_JSON=json.dumps(needs)),
            ).returncode

        self.assertEqual(execute(base), 0)
        for key in base:
            for state in ("failure", "cancelled", "skipped"):
                with self.subTest(job=key, state=state):
                    mutated = {**base, key: {**base[key], "result": state}}
                    self.assertNotEqual(execute(mutated), 0)
        unrelated = {k: {"result": "skipped"} for k in base}
        unrelated["check-changes"] = {"result": "success", "outputs": {"ppu": "false"}}
        self.assertEqual(execute(unrelated), 0)

    def test_finish_requires_both_current_receipts(self):
        import json
        import os
        import tempfile

        steps = self.load("pr-test-ppu.yml")["jobs"]["pr-test-ppu-finish"]["steps"]
        step = next(s for s in steps if s.get("id") == "receipts")
        download = next(
            s for s in steps if s.get("uses") == "actions/download-artifact@v4"
        )
        self.assertEqual(
            download["with"]["pattern"], "ppu-smoke-*-${{ github.run_attempt }}-receipt"
        )
        identity = dict(
            git_sha="a" * 40,
            build_image="sdk@sha256:" + "b" * 64,
            manifest_sha256="c" * 64,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            env = dict(
                os.environ,
                RECEIPT_ROOT=temp,
                TESTED_SHA=identity["git_sha"],
                BUILD_IMAGE=identity["build_image"],
                MANIFEST_SHA256=identity["manifest_sha256"],
            )

            def execute():
                return subprocess.run(
                    ["bash", "-e", "-c", step["run"]],
                    env=env,
                    capture_output=True,
                    text=True,
                ).returncode

            self.assertNotEqual(execute(), 0)
            for board in ("ZW810E", "ZW-M890P"):
                (root / (board + ".json")).write_text(
                    json.dumps(dict(identity, expected_device=board))
                )
                if board == "ZW810E":
                    self.assertNotEqual(execute(), 0)
            self.assertEqual(execute(), 0)
            for key in ("git_sha", "build_image", "manifest_sha256", "expected_device"):
                with self.subTest(key=key):
                    bad = dict(identity, expected_device="ZW-M890P", **{})
                    bad[key] = "wrong"
                    (root / "ZW-M890P.json").write_text(json.dumps(bad))
                    self.assertNotEqual(execute(), 0)

    def test_artifact_identity_wiring(self):
        main = self.load("pr-test-ppu.yml")["jobs"]
        for key in ("tested_sha", "build_image"):
            expected = "${{ needs.check-changes.outputs." + key + " }}"
            for job in ("ppu-wheel-build", "ppu-smoke"):
                self.assertEqual(main[job]["with"][key], expected)
        self.assertEqual(
            main["ppu-smoke"]["with"]["manifest_sha256"],
            "${{ needs.ppu-wheel-build.outputs.manifest_sha256 }}",
        )
        jobs = self.load("_pr-test-ppu-build.yml")["jobs"]
        self.assertIn("runtime", jobs)
        runtime = jobs["runtime"]
        smoke = self.load("_pr-test-ppu-smoke.yml")["jobs"]["smoke"]
        upload = next(
            s for s in runtime["steps"] if s.get("uses") == "actions/upload-artifact@v4"
        )
        download = next(
            s for s in smoke["steps"] if s.get("uses") == "actions/download-artifact@v4"
        )
        self.assertEqual(upload["with"]["name"], download["with"]["name"])
        self.assertEqual(upload["with"]["if-no-files-found"], "error")
        self.assertEqual(
            smoke["container"]["image"], "ghcr.io/actions/actions-runner:latest"
        )
        scheduler = next(s for s in smoke["steps"] if s.get("id") == "scheduler")
        self.assertEqual(scheduler["with"]["image"], "${{ inputs.build_image }}")
        self.assertEqual(smoke["env"]["SAIL_BUILD_IMAGE"], "${{ inputs.build_image }}")
        commands = "\n".join(s.get("run", "") for s in smoke["steps"])
        self.assertNotIn("prepare_sail_env.py", commands)
        self.assertNotIn("import torch", commands)
        self.assertEqual(
            smoke["env"]["SAIL_MANIFEST_SHA256"], "${{ inputs.manifest_sha256 }}"
        )

    def test_suite_budget_and_wrapper(self):
        script = TOOLS / "run_sail_smoke.sh"
        self.assertTrue(script.is_file(), "smoke wrapper is missing")
        source = script.read_text()
        for token in (
            "900s",
            "2100s",
            "--timeout-from-est-time",
            "SGLANG_TEST_MAX_RETRY=0",
            "HF_HUB_OFFLINE=0",
            "registration",
            "result.json",
            "mktemp -d",
            "trap",
        ):
            self.assertIn(token, source)
        self.assertNotIn("--enable-retry", source)
        tree = ast.parse((ROOT / "python/sglang/test/ci/ci_utils.py").read_text())
        nodes = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "derive_timeout_per_file"
            or isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Name) and t.id.startswith("DERIVED_TIMEOUT_")
                for t in n.targets
            )
        ]
        scope = {}
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "<timeout>", "exec"), scope
        )
        self.assertEqual(scope["derive_timeout_per_file"](300), 2100)
        self.assertIn(
            '"stage-a-test-1-gpu-ppu"', (ROOT / "test/run_suite.py").read_text()
        )


if __name__ == "__main__":
    unittest.main()
