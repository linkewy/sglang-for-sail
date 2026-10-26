from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention import attention_registry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSailAttentionRegistry(CustomTestCase):
    def test_fa3_rejects_mla_with_actionable_flashmla_hint(self):
        runner = SimpleNamespace(use_mla_backend=True)
        with (
            patch.object(attention_registry, "_is_sail", True),
            patch.object(attention_registry, "_is_musa", False),
            patch.object(
                attention_registry, "get_device_capability", return_value=(8, 0)
            ),
            self.assertRaisesRegex(ValueError, "Use --attention-backend flashmla"),
        ):
            attention_registry.create_flashattention_v3_backend(runner)


if __name__ == "__main__":
    import unittest

    unittest.main()
