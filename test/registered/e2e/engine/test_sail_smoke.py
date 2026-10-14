"""Exercise SAIL platform selection and real HTTP inference on a single PPU."""

import os
import unittest

import requests
import torch

from sglang.srt.platforms import current_platform
from sglang.test.ci.ci_register import register_ppu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_ppu_ci(est_time=300, suite="stage-a-test-1-gpu-ppu")


class TestSailSmoke(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.process = None
        assert current_platform.is_sail(), "PPU smoke requires the SAIL platform"
        assert torch.cuda.device_count() == 1, "PPU smoke requires exactly one PPU"

        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            os.environ.get("PPU_CI_MODEL_PATH") or "Qwen/Qwen3-0.6B",
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--attention-backend",
                "triton",
                "--disable-prefill-cuda-graph",
                "--disable-decode-cuda-graph",
                "--tp-size",
                "1",
                "--context-length",
                "1024",
                "--mem-fraction-static",
                "0.2",
                "--max-running-requests",
                "4",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def test_health_and_generate(self):
        health = requests.get(f"{self.base_url}/health", timeout=30)
        self.assertEqual(health.status_code, 200, health.text)

        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": "The capital of France is",
                "sampling_params": {"temperature": 0, "max_new_tokens": 32},
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200, response.text)
        output = response.json()
        self.assertTrue(output["text"].strip(), output)
        self.assertGreater(output["meta_info"]["completion_tokens"], 0, output)


if __name__ == "__main__":
    unittest.main()
