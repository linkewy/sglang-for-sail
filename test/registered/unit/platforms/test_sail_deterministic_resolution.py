import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

from sglang.srt.arg_groups import platform_hook
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _fallback_view(**overrides):
    values = {
        "attention_backend": "fa3",
        "attn_cp_size": 1,
        "cuda_graph_config": object(),
        "dcp_size": 1,
        "enable_deterministic_inference": True,
        "enable_kv_cache_sharding": False,
        "enable_prefill_cp": False,
        "enable_unified_memory": False,
        "kv_cache_dtype": "auto",
        "pp_size": 1,
        "speculative_algorithm": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _supported_model_config(**overrides):
    values = {
        "is_encoder_decoder": False,
        "is_local_attention_model": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _load_sail_backend_module():
    interface = ModuleType("flash_attn_interface")
    interface.flash_attn_varlen_func = Mock()
    interface.flash_attn_with_kvcache = Mock()
    interface.get_scheduler_metadata = Mock()
    with patch.dict(sys.modules, {"flash_attn_interface": interface}):
        from sglang.srt.hardware_backend.sail.attention import (
            flashattention_backend as sail_backend,
        )

    return sail_backend


class TestSailDeterministicResolution(CustomTestCase):
    def test_fa3_prefill_uses_serial_eager_no_cache_mode(self):
        graph = object()
        resolved_graph = object()
        view = _fallback_view(cuda_graph_config=graph)
        with (
            patch.object(
                platform_hook,
                "get_platform",
                return_value=SimpleNamespace(is_sail=True),
            ),
            patch.object(
                platform_hook,
                "attention_backends_of",
                return_value=("fa3", "fa3"),
            ),
            patch.object(platform_hook, "use_mla_backend", return_value=False),
            patch.object(
                platform_hook,
                "model_config_of",
                return_value=_supported_model_config(),
            ),
            patch.object(
                platform_hook, "with_phase", return_value=resolved_graph
            ) as with_phase,
        ):
            result = platform_hook._sail_fa3_deterministic_fallback(view)

        self.assertEqual(
            result,
            {
                "cuda_graph_config": resolved_graph,
                "chunked_prefill_size": -1,
                "disable_overlap_schedule": True,
                "disable_radix_cache": True,
                "enable_dynamic_chunking": False,
                "enable_mixed_chunk": False,
                "prefill_max_requests": 1,
            },
        )
        with_phase.assert_called_once_with(
            graph,
            platform_hook.Phase.PREFILL,
            backend=platform_hook.Backend.DISABLED,
        )

    def test_unsupported_fa3_deterministic_configurations_are_rejected(self):
        cases = (
            ({"kv_cache_dtype": "fp8_e4m3"}, {}, "kv_cache_dtype"),
            ({"enable_prefill_cp": True}, {}, "context parallelism"),
            ({"enable_unified_memory": True}, {}, "unified memory"),
            ({"enable_kv_cache_sharding": True}, {}, "KV-cache sharding"),
            ({"speculative_algorithm": "EAGLE"}, {}, "speculative decoding"),
            ({"pp_size": 2}, {}, "pipeline parallelism"),
            ({}, {"is_local_attention_model": True}, "local attention"),
            ({}, {"is_encoder_decoder": True}, "cross attention"),
        )
        for view_overrides, model_overrides, message in cases:
            with self.subTest(message=message):
                with (
                    patch.object(
                        platform_hook,
                        "get_platform",
                        return_value=SimpleNamespace(is_sail=True),
                    ),
                    patch.object(
                        platform_hook,
                        "attention_backends_of",
                        return_value=("fa3", "fa3"),
                    ),
                    patch.object(platform_hook, "use_mla_backend", return_value=False),
                    patch.object(
                        platform_hook,
                        "model_config_of",
                        return_value=_supported_model_config(**model_overrides),
                    ),
                ):
                    with self.assertRaisesRegex(ValueError, message):
                        platform_hook._sail_fa3_deterministic_fallback(
                            _fallback_view(**view_overrides)
                        )

    def test_early_handler_declares_platform_constraints_directly(self):
        server_args = object()
        fallback = {
            "cuda_graph_config": object(),
            "chunked_prefill_size": -1,
            "disable_overlap_schedule": True,
            "disable_radix_cache": True,
            "enable_dynamic_chunking": False,
            "enable_mixed_chunk": False,
            "prefill_max_requests": 1,
        }
        resolved = SimpleNamespace()
        with (
            patch.object(
                platform_hook,
                "get_platform",
                return_value=SimpleNamespace(is_sail=True),
            ),
            patch.object(platform_hook, "resolved_view", return_value=resolved),
            patch.object(
                platform_hook,
                "attention_backends_of",
                return_value=("fa3", "fa3"),
            ),
            patch.object(
                platform_hook,
                "_sail_fa3_deterministic_fallback",
                return_value=fallback,
            ) as deterministic_fallback,
            patch.object(platform_hook, "declare_resolution") as declare_resolution,
        ):
            platform_hook.handle_sail_deterministic_fa3_prefill(server_args)

        deterministic_fallback.assert_called_once_with(resolved, prefill_backend="fa3")
        declare_resolution.assert_called_once_with(
            server_args,
            "_sail_fa3_deterministic_fallback",
            **fallback,
        )

    def test_early_handler_does_not_guess_an_unresolved_backend(self):
        server_args = object()
        view = _fallback_view(attention_backend=None)
        with (
            patch.object(
                platform_hook,
                "get_platform",
                return_value=SimpleNamespace(is_sail=True),
            ),
            patch.object(platform_hook, "resolved_view", return_value=view),
            patch.object(
                platform_hook,
                "attention_backends_of",
                return_value=(None, None),
            ),
            patch.object(platform_hook, "declare_resolution") as declare_resolution,
        ):
            platform_hook.handle_sail_deterministic_fa3_prefill(server_args)

        declare_resolution.assert_not_called()

    def test_late_fa3_without_pre_memory_constraints_is_rejected(self):
        server_args = SimpleNamespace(_resolved_overrides=[])
        resolved = SimpleNamespace(
            attention_backend="fa3",
            prefill_attention_backend="fa3",
            decode_attention_backend="fa3",
            enable_deterministic_inference=True,
            speculative_draft_attention_backend=None,
        )
        with (
            patch.object(
                platform_hook,
                "get_platform",
                return_value=SimpleNamespace(is_sail=True),
            ),
            patch.object(platform_hook, "resolved_view", return_value=resolved),
            patch.object(platform_hook, "run_post_process_pass"),
        ):
            with self.assertRaisesRegex(ValueError, "selected before memory sizing"):
                platform_hook.handle_sail_attention_backend_compatibility(server_args)

    def test_late_backend_change_after_fa3_constraints_is_rejected(self):
        server_args = SimpleNamespace(
            _resolved_overrides=[("_sail_fa3_deterministic_fallback", {})]
        )
        resolved = SimpleNamespace(
            attention_backend="triton",
            prefill_attention_backend="triton",
            decode_attention_backend="triton",
            enable_deterministic_inference=True,
            speculative_draft_attention_backend=None,
        )
        with (
            patch.object(
                platform_hook,
                "get_platform",
                return_value=SimpleNamespace(is_sail=True),
            ),
            patch.object(platform_hook, "resolved_view", return_value=resolved),
            patch.object(platform_hook, "run_post_process_pass"),
        ):
            with self.assertRaisesRegex(ValueError, "changed to 'triton'"):
                platform_hook.handle_sail_attention_backend_compatibility(server_args)

    def test_non_sail_or_non_deterministic_configuration_is_unchanged(self):
        for is_sail, deterministic, prefill_backend, is_mla in (
            (False, True, "fa3", False),
            (True, False, "fa3", False),
            (True, True, "triton", False),
            (True, True, "fa3", True),
        ):
            with self.subTest(
                is_sail=is_sail,
                deterministic=deterministic,
                prefill_backend=prefill_backend,
                is_mla=is_mla,
            ):
                view = _fallback_view(
                    attention_backend=prefill_backend,
                    enable_deterministic_inference=deterministic,
                )
                with (
                    patch.object(
                        platform_hook,
                        "get_platform",
                        return_value=SimpleNamespace(is_sail=is_sail),
                    ),
                    patch.object(
                        platform_hook,
                        "attention_backends_of",
                        return_value=(prefill_backend, prefill_backend),
                    ),
                    patch.object(platform_hook, "use_mla_backend", return_value=is_mla),
                ):
                    self.assertEqual(
                        platform_hook._sail_fa3_deterministic_fallback(view), {}
                    )

    def test_raw_varlen_dispatch_preserves_platform_kernel_and_lse(self):
        backend = object.__new__(FlashAttentionBackend)
        backend.flash_attn_varlen_func = Mock(return_value="platform")
        with patch.object(
            backend, "_call_flash_attn_varlen_func", return_value="deterministic"
        ) as deterministic_call:
            self.assertEqual(
                backend._call_raw_varlen_prefill(
                    deterministic=False, return_softmax_lse=True
                ),
                "platform",
            )
            backend.flash_attn_varlen_func.assert_called_once_with(
                return_softmax_lse=True
            )
            deterministic_call.assert_not_called()

            self.assertEqual(
                backend._call_raw_varlen_prefill(
                    deterministic=True, return_softmax_lse=True
                ),
                "deterministic",
            )
            deterministic_call.assert_called_once_with(return_softmax_lse=True)

    def test_sail_runtime_predicate_accepts_only_cold_extend(self):
        sail_backend = _load_sail_backend_module()
        backend = object.__new__(sail_backend.SailFlashAttentionBackend)
        backend.use_mla = False
        backend.has_local_attention = False
        backend.kv_index_translator = SimpleNamespace(is_translating=False)
        backend._kv_shard_pool = None
        backend.kv_cache_dtype_str = "auto"
        forward_batch = SimpleNamespace(
            extend_prefix_lens_cpu=[0],
            forward_mode=ForwardMode.EXTEND,
        )
        layer = SimpleNamespace(is_cross_attention=False)
        deterministic = SimpleNamespace(enable_deterministic_inference=True)
        with patch.object(
            sail_backend,
            "get_exec",
            return_value=SimpleNamespace(deterministic=deterministic),
        ):
            self.assertTrue(
                backend._use_deterministic_varlen_prefill(
                    forward_batch, layer, object(), object(), False
                )
            )
            forward_batch.extend_prefix_lens_cpu = [1]
            with self.assertRaisesRegex(RuntimeError, "non-cold prefill"):
                backend._use_deterministic_varlen_prefill(
                    forward_batch, layer, object(), object(), False
                )
            forward_batch.extend_prefix_lens_cpu = [0]
            with self.assertRaisesRegex(RuntimeError, "context parallelism"):
                backend._use_deterministic_varlen_prefill(
                    forward_batch, layer, object(), object(), True
                )


if __name__ == "__main__":
    import unittest

    unittest.main()
