import hashlib
import importlib.metadata as metadata
import json
import os
import sys
import unittest
from pathlib import Path

import psutil
import requests
import torch
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import popen_launch_server, terminate_and_kill_process_tree

register_ppu_ci(est_time=300, stage="stage-a", runner_config="1-gpu-ppu")


class TestSailQwen3Smoke(unittest.TestCase):
    evidence = {"device": "", "http": {}, "native": {}, "cleanup_ok": False}

    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one real PPU is required")
        device = torch.cuda.get_device_name(0)
        expected = os.environ["SGLANG_PPU_EXPECTED_DEVICE_NAME"]
        if expected not in ("ZW810E", "ZW-M890P") or expected not in device:
            raise RuntimeError(f"Unexpected PPU: {device}")
        cls.evidence["device"] = device
        model = Path(os.environ["SGLANG_PPU_SMOKE_MODEL"])
        if (
            not model.is_absolute()
            or not model.is_dir()
            or not (model / "config.json").is_file()
        ):
            raise RuntimeError("Smoke requires an existing local Qwen3-0.6B checkpoint")
        cls.base_url = "http://127.0.0.1:30000"
        logs = Path(os.environ["SAIL_SMOKE_RESULT_DIR"])
        stdout = (logs / "server.stdout.log").open("x")
        cls.addClassCleanup(stdout.close)
        stderr = (logs / "server.stderr.log").open("x")
        cls.addClassCleanup(stderr.close)
        cls.process = popen_launch_server(
            str(model),
            cls.base_url,
            timeout=600,
            other_args=[
                "--attention-backend",
                "triton",
                "--sampling-backend",
                "pytorch",
                "--cuda-graph-backend-prefill",
                "disabled",
                "--cuda-graph-backend-decode",
                "disabled",
            ],
            env={"HF_HUB_OFFLINE": "0", "SGLANG_TEST_MAX_RETRY": "0"},
            return_stdout_stderr=(stdout, stderr),
        )
        cls.addClassCleanup(cls.cleanup_server)

    @classmethod
    def cleanup_server(cls):
        parent = psutil.Process(cls.process.pid)
        processes = [parent, *parent.children(recursive=True)]
        terminate_and_kill_process_tree(cls.process, terminate_timeout=30)
        _, alive = psutil.wait_procs(processes, timeout=30)
        if alive:
            raise RuntimeError("Server process tree survived cleanup")
        cls.evidence["cleanup_ok"] = True

    def test_generate(self):
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        text = payload.get("text")
        tokens = payload.get("meta_info", {}).get("completion_tokens")
        self.assertIsInstance(text, str)
        self.assertTrue(text.strip())
        self.assertIs(type(tokens), int)
        self.assertGreater(tokens, 0)
        self.evidence["http"] = {
            "status": 200,
            "text_nonempty": True,
            "completion_tokens": tokens,
        }

    def test_kernel_ops(self):
        import sgl_kernel
        from sgl_kernel import common_ops
        from sgl_kernel.sampling import top_k_renorm_probs, top_p_renorm_probs

        dist = metadata.distribution("sglang-kernel")
        prefix = Path(sys.prefix).resolve()
        package = Path(sgl_kernel.__file__).resolve()
        native = Path(common_ops.__file__).resolve()
        self.assertTrue(
            package.is_relative_to(prefix) and "site-packages" in package.parts
        )
        self.assertTrue(
            native.is_relative_to(prefix) and "site-packages" in native.parts
        )
        relative = native.relative_to(Path(dist.locate_file("")).resolve()).as_posix()
        artifact = Path(os.environ["SAIL_ARTIFACT_DIR"])
        manifest = json.loads((artifact / "manifest.json").read_text())
        kernel = next(w for w in manifest["wheels"] if w["name"] == "sglang-kernel")
        with native.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        self.assertEqual(kernel["native_files"][relative], digest)
        self.evidence["native"] = {"relative_path": relative, "sha256": digest}

        torch.manual_seed(42)
        x = torch.randn(19, 1024, dtype=torch.float16, device="cuda")
        weight = torch.randn(1024, dtype=torch.float16, device="cuda")
        residual = torch.randn_like(x)

        def norm(value, gemma=False):
            value = value.float()
            value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + 1e-6)
            return (value * (weight.float() + (1.0 if gemma else 0.0))).to(x.dtype)

        def close(actual, expected):
            torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)

        close(sgl_kernel.rmsnorm(x, weight, enable_pdl=False), norm(x))
        close(sgl_kernel.gemma_rmsnorm(x, weight, enable_pdl=False), norm(x, True))
        for gemma, operation in (
            (False, sgl_kernel.fused_add_rmsnorm),
            (True, sgl_kernel.gemma_fused_add_rmsnorm),
        ):
            with self.subTest(gemma=gemma):
                actual, actual_residual = x.clone(), residual.clone()
                summed = x + residual if gemma else x.float() + residual.float()
                operation(actual, actual_residual, weight, 1e-6, enable_pdl=False)
                close(actual, norm(summed, gemma))
                close(actual_residual, summed.to(x.dtype))
        probs = torch.rand(3, 111, device="cuda", dtype=torch.float32)
        probs /= probs.sum(-1, keepdim=True)
        for k in (10, 100):
            pivot = probs.sort(descending=True).values[:, k - 1 : k]
            expected = probs * (probs >= pivot)
            expected /= expected.sum(-1, keepdim=True)
            close(top_k_renorm_probs(probs, k), expected)
        for p in (0.1, 0.5, 0.9):
            ordered, indices = probs.sort(descending=False)
            mask = torch.zeros_like(probs).scatter_(
                1, indices, (ordered.cumsum(-1) >= 1 - p).float()
            )
            expected = probs * mask
            expected /= expected.sum(-1, keepdim=True)
            close(top_p_renorm_probs(probs, p), expected)
        torch.cuda.synchronize()


class SmokeResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.successful_tests = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.successful_tests.append(test._testMethodName)


if __name__ == "__main__":
    result = unittest.main(
        exit=False, testRunner=unittest.TextTestRunner(resultclass=SmokeResult)
    ).result
    document = dict(
        TestSailQwen3Smoke.evidence,
        schema_version=1,
        run_token=os.environ["SAIL_SMOKE_RUN_TOKEN"],
        git_sha=os.environ["SAIL_TESTED_SHA"],
        manifest_sha256=os.environ["SAIL_MANIFEST_SHA256"],
        expected_device=os.environ["SGLANG_PPU_EXPECTED_DEVICE_NAME"],
        tests_run=result.testsRun,
        failures=len(result.failures),
        errors=len(result.errors),
        skipped=len(result.skipped),
        successful_tests=result.successful_tests,
    )
    output = Path(os.environ["SAIL_SMOKE_RESULT_DIR"]) / "result.json"
    with output.open("x") as stream:
        json.dump(document, stream, indent=2, sort_keys=True)
    sys.exit(
        0
        if result.wasSuccessful() and result.testsRun == 2 and not result.skipped
        else 1
    )
