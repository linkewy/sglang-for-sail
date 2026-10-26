from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.attention import attention_registry
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestSailAttentionRegistry(CustomTestCase):
    @staticmethod
    def _draft_factory():
        from sglang.srt.speculative.draft_utils import DraftBackendFactory

        factory = object.__new__(DraftBackendFactory)
        factory.draft_model_runner = object()
        factory.topk = 1
        factory.speculative_num_steps = 3
        return factory

    def test_draft_factory_selects_sail_fa3_and_flashmla_backends(self):
        import sys
        from types import ModuleType
        from unittest.mock import Mock

        from sglang.srt.speculative import draft_utils

        factory = self._draft_factory()
        fa_prefill = object()
        fa_decode = object()
        mla_prefill = object()
        mla_decode = object()

        fa_module_name = (
            "sglang.srt.hardware_backend.sail.attention.flashattention_backend"
        )
        fa_module = ModuleType(fa_module_name)
        fa_module.SailFlashAttentionBackend = Mock(return_value=fa_prefill)
        fa_module.SailFlashAttentionMultiStepBackend = Mock(return_value=fa_decode)

        mla_module_name = "sglang.srt.hardware_backend.sail.attention.flashmla_backend"
        mla_module = ModuleType(mla_module_name)
        mla_module.SailFlashMLABackend = Mock(return_value=mla_prefill)
        mla_module.SailFlashMLAMultiStepDraftBackend = Mock(return_value=mla_decode)

        with (
            patch.object(draft_utils, "is_sail", return_value=True),
            patch.dict(
                sys.modules,
                {fa_module_name: fa_module, mla_module_name: mla_module},
            ),
        ):
            self.assertEqual(factory._create_fa3_prefill_backend(), ("fa3", fa_prefill))
            self.assertEqual(factory._create_fa3_decode_backend(), ("fa3", fa_decode))
            self.assertEqual(
                factory._create_flashmla_prefill_backend(),
                ("flashmla", mla_prefill),
            )
            self.assertEqual(
                factory._create_flashmla_decode_backend(),
                ("flashmla", mla_decode),
            )

        fa_module.SailFlashAttentionBackend.assert_called_once_with(
            factory.draft_model_runner, skip_prefill=False, fa_impl_ver=3
        )
        fa_module.SailFlashAttentionMultiStepBackend.assert_called_once_with(
            factory.draft_model_runner, 1, 3, fa_impl_ver=3
        )
        mla_module.SailFlashMLABackend.assert_called_once_with(
            factory.draft_model_runner, skip_prefill=False
        )
        mla_module.SailFlashMLAMultiStepDraftBackend.assert_called_once_with(
            factory.draft_model_runner, 1, 3
        )

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
