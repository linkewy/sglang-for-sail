"""SAIL FlashMLA backend selected by the in-tree backend registry."""

import logging
from typing import TYPE_CHECKING, Optional

import torch
import triton

from sglang.kernels.ops.attention.kv_indices import (
    create_flashmla_kv_indices_triton,
    get_num_kv_index_blocks_flashmla,
)
from sglang.srt.hardware_backend.sail.attention.flash_mla import get_mla_metadata
from sglang.srt.layers.attention.flashmla_backend import (
    PAGE_SIZE,
    FlashMLABackend,
    FlashMLADecodeMetadata,
)
from sglang.srt.model_executor.forward_batch_info import ForwardMode

if TYPE_CHECKING:
    from sglang.srt.mem_cache.kv_loc_plan import KVLocPlan
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)
_SAIL_MAX_SM_PARTS = 320


class SailFlashMLABackend(FlashMLABackend):
    """FlashMLA backend with SAIL's lazy scheduler-metadata semantics."""

    def __init__(self, model_runner: "ModelRunner", **kwargs):
        super().__init__(model_runner, **kwargs)
        if self.is_fp8_kvcache:
            raise NotImplementedError(
                "SAIL FlashMLA does not support dense FP8 KV cache; use BF16 KV cache."
            )

    def init_cuda_graph_state(self, *args, **kwargs):
        super().init_cuda_graph_state(*args, **kwargs)
        buf = self.cuda_graph_mla_metadata
        if buf is not None and buf.shape[0] > _SAIL_MAX_SM_PARTS:
            self.cuda_graph_mla_metadata = torch.empty(
                (_SAIL_MAX_SM_PARTS, 8),
                dtype=torch.int32,
                device=buf.device,
            )

    def _apply_decode_target_verify_metadata(
        self,
        plan: "KVLocPlan",
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: Optional[torch.Tensor],
        forward_mode: ForwardMode,
    ):
        seq_lens = seq_lens[:bs]
        seq_lens_cpu = seq_lens_cpu[:bs] if seq_lens_cpu is not None else None

        if forward_mode.is_target_verify():
            seq_lens = seq_lens + self.num_draft_tokens
            if seq_lens_cpu is not None:
                seq_lens_cpu = seq_lens_cpu + self.num_draft_tokens

        max_seqlen_pad = (
            triton.cdiv(seq_lens_cpu.max().item(), PAGE_SIZE)
            if seq_lens_cpu is not None
            else self.cuda_graph_kv_indices.shape[1]
        )

        if self.kv_index_translator.reads_are_translated:
            assert self.page_size == PAGE_SIZE
            self.kv_index_translator.copy_page_table(
                plan, out=self.cuda_graph_kv_indices[:bs]
            )
        else:
            create_flashmla_kv_indices_triton[
                (
                    bs,
                    get_num_kv_index_blocks_flashmla(
                        self.cuda_graph_kv_indices.stride(0), PAGE_SIZE
                    ),
                )
            ](
                self.req_to_token,
                req_pool_indices[:bs],
                seq_lens,
                None,
                self.cuda_graph_kv_indices,
                self.req_to_token.stride(0),
                self.cuda_graph_kv_indices.stride(0),
            )

        q_head_mult = (
            self.num_draft_tokens
            if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2()
            else 1
        )
        scheduler_metadata, _ = get_mla_metadata(
            seq_lens.to(torch.int32),
            q_head_mult * self.num_q_heads,
            1,
            is_fp8_kvcache=False,
        )

        seq_lens_k = None
        if forward_mode.is_draft_extend_v2():
            self.cuda_graph_draft_extend_seq_lens_k[:bs].copy_(seq_lens)
            seq_lens_k = self.cuda_graph_draft_extend_seq_lens_k[:bs]

        self.forward_metadata = FlashMLADecodeMetadata(
            scheduler_metadata,
            None,
            self.cuda_graph_kv_indices[:bs, :max_seqlen_pad],
            seq_lens_k if seq_lens_k is not None else seq_lens.to(torch.int32),
        )

    def init_forward_metadata_in_graph(self, forward_batch: "ForwardBatch") -> None:
        metadata = self.forward_metadata
        if isinstance(metadata, FlashMLADecodeMetadata):
            assert metadata.flashmla_metadata is not None
            metadata.flashmla_metadata.have_initialized = False
