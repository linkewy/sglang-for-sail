"""SAIL FlashAttention backend selected by the in-tree backend registry."""

from typing import TYPE_CHECKING

from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


class SailFlashAttentionBackend(FlashAttentionBackend):
    """FlashAttention backend using T-HEAD's SAIL FA3 interface."""

    def __init__(self, model_runner: "ModelRunner", **kwargs):
        super().__init__(model_runner, **kwargs)
        from sglang.srt.hardware_backend.sail.attention.flash_attention import (
            flash_attn_varlen_func,
            flash_attn_with_kvcache,
            get_scheduler_metadata,
        )

        self.flash_attn_varlen_func = flash_attn_varlen_func
        self.flash_attn_with_kvcache = flash_attn_with_kvcache
        self._get_scheduler_metadata = get_scheduler_metadata
