from __future__ import annotations

from typing import TYPE_CHECKING

from sglang.srt.hardware_backend.sail.attention.flash_attention import (
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    get_scheduler_metadata,
)
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionBackend,
    FlashAttentionMultiStepBackend,
)

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.runtime_context import get_exec

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class SailFlashAttentionBackend(FlashAttentionBackend):
    """FlashAttention backend running on the SAIL FA3 build."""

    def _init_flash_attention_functions(self, device_capability):
        if self.fa_impl_ver != 3:
            raise NotImplementedError(
                f"FA{self.fa_impl_ver} is not supported on SAIL; use fa3."
            )
        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.flash_attn_with_kvcache = flash_attn_with_kvcache
        self._get_scheduler_metadata = get_scheduler_metadata
        self._get_fa_runtime_policy = None

    def _call_flash_attn_with_kvcache(self, **kwargs):
        return flash_attn_with_kvcache(**kwargs)

    def _call_flash_attn_varlen_func(self, **kwargs):
        if get_exec().deterministic.enable_deterministic_inference:
            kwargs["deterministic"] = True
            kwargs.setdefault("pack_gqa", False)
        return flash_attn_varlen_func(**kwargs)

    def _use_deterministic_varlen_prefill(
        self, forward_batch, layer, k, v, cp_active: bool
    ) -> bool:
        """Use the vendor's batch-invariant varlen path for cold prefill."""
        if not get_exec().deterministic.enable_deterministic_inference:
            return False

        prefix_lens = forward_batch.extend_prefix_lens_cpu
        unsupported = []
        if cp_active:
            unsupported.append("context parallelism")
        if forward_batch.forward_mode != ForwardMode.EXTEND:
            unsupported.append(f"forward_mode={forward_batch.forward_mode}")
        if prefix_lens is None or any(prefix_lens):
            unsupported.append("non-cold prefill")
        if k is None or v is None:
            unsupported.append("missing raw K/V")
        if self.use_mla:
            unsupported.append("MLA")
        if layer.is_cross_attention:
            unsupported.append("cross attention")
        if self.has_local_attention:
            unsupported.append("local attention")
        if self.kv_index_translator.is_translating:
            unsupported.append("translated KV indices")
        if self._kv_shard_pool is not None:
            unsupported.append("KV-cache sharding")
        if self.kv_cache_dtype_str != "auto":
            unsupported.append(f"kv_cache_dtype={self.kv_cache_dtype_str}")
        if unsupported:
            raise RuntimeError(
                "SAIL FA3 deterministic varlen prefill invariant violated: "
                + ", ".join(unsupported)
            )
        return True


class SailFlashAttentionMultiStepBackend(FlashAttentionMultiStepBackend):
    def __init__(
        self,
        model_runner: ModelRunner,
        topk: int,
        speculative_num_steps: int,
        fa_impl_ver: int = 3,
    ):
        self.model_runner = model_runner
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.kv_index_translator = model_runner.kv_index_translator
        self.attn_backends = []
        for i in range(self.speculative_num_steps - 1):
            self.attn_backends.append(
                SailFlashAttentionBackend(
                    model_runner,
                    speculative_step_id=i,
                    topk=self.topk,
                    speculative_num_steps=self.speculative_num_steps,
                    fa_impl_ver=fa_impl_ver,
                )
            )
        self.attn_backend_list = self.attn_backends
        self.forward_metadata = None
