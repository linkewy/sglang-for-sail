# SPDX-License-Identifier: Apache-2.0
"""Step5 sparse decode over native FULL KV in portable Triton (SM90 replacement).

Mirrors ``cute/sm90/decode_sm90_gqa.py`` for devices that cannot run the CuTe DSL
kernel (e.g. sm80 PPU). Grid is ``(row, head)``; each program owns one query row
and runs an online softmax over that row's own packed region list, writing the
final BF16 output and FP32 natural LSE directly.

The SM90 split-KV path (1/2/4-way partitions plus a separate natural-LSE merge)
is bypassed entirely here: every row is computed in a single pass. This trades
decode-batch occupancy for simplicity and is correctness-first, not yet tuned.

Each row's ``packed_regions[row, win]`` is an int64 whose high word is the
physical region id and low word the logical start token::

    phys_region = packed >> 32          # -> K/V slot region
    start_tok   = packed & 0xFFFFFFFF    # -> logical position of lane 0
    phys_tok    = phys_region * 8 + lane
    logical_tok = start_tok + lane

A key is usable only when all hold, matching the SM90 ``_mask_scores``:

1. ``packed >= 0``
2. physical slot valid: ``0 <= phys_region < slots // 8``
3. within the row's selection: ``tok < counts[row] * 8``
4. causal: ``logical_tok < causal_lengths[row]``

GQA has a single KV head, so all logical heads share K/V. Rows at or beyond
``valid_rows[0]`` (or with zero counts) write zero output and ``-inf`` LSE.
Only BF16 K/V is supported; the FP8 paged path is not implemented.
"""

import torch
import triton
import triton.language as tl

REGION_TOKENS = tl.constexpr(8)


@triton.jit(
    do_not_specialize=("ROWS", "MAX_PHYS_REGIONS", "VALID_ROWS"),
    do_not_specialize_on_alignment=("ROWS", "MAX_PHYS_REGIONS", "VALID_ROWS"),
)
def _sparse_decode(
    Q,
    K,
    V,
    PACKED,
    COUNTS,
    CAUSAL_LEN,
    VALID_ROWS,
    OUT,
    LSE,
    ROWS,
    MAX_PHYS_REGIONS,
    SOFTMAX_SCALE,
    HEADS: tl.constexpr,
    PACKED_WIDTH: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)

    valid_row = row < tl.load(VALID_ROWS)
    count = tl.load(COUNTS + row, mask=valid_row, other=0)
    windows = tl.minimum(count, PACKED_WIDTH)
    count_tokens = windows * REGION_TOKENS
    causal_len = tl.maximum(tl.load(CAUSAL_LEN + row, mask=valid_row, other=0), 0)

    channel = tl.arange(0, 256)
    cmask = channel < 192

    q_vec = tl.load(
        Q + row * (HEADS * 192) + head * 192 + channel,
        mask=cmask & valid_row,
        other=0.0,
    ).to(tl.float32)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros((256,), dtype=tl.float32)

    for start in range(0, count_tokens, BLOCK_N):
        tok = start + tl.arange(0, BLOCK_N)
        tvalid = tok < count_tokens
        win = tok // REGION_TOKENS
        lane = tok - win * REGION_TOKENS

        packed = tl.load(PACKED + row * PACKED_WIDTH + win, mask=tvalid, other=-1)
        phys_region = (packed >> 32).to(tl.int32)
        start_tok = (packed & 0xFFFFFFFF).to(tl.int32)
        logical_tok = start_tok + lane
        phys_tok = (phys_region.to(tl.int64) * REGION_TOKENS + lane)

        keep = (
            tvalid
            & (packed >= 0)
            & (phys_region >= 0)
            & (phys_region < MAX_PHYS_REGIONS)
            & (logical_tok < causal_len)
        )

        kmask = keep[:, None] & cmask[None, :]
        k = tl.load(K + phys_tok[:, None] * 192 + channel[None, :], mask=kmask, other=0.0).to(tl.float32)
        score = tl.sum(k * q_vec[None, :], axis=1) * SOFTMAX_SCALE
        score = tl.where(keep, score, float("-inf"))
        v = tl.load(V + phys_tok[:, None] * 192 + channel[None, :], mask=kmask, other=0.0).to(tl.float32)

        m_new = tl.maximum(m_i, tl.max(score, axis=0))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.exp(score - m_safe)
        alpha = tl.exp(m_i - m_safe)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        m_i = m_new

    has = l_i > 0.0
    l_safe = tl.where(has, l_i, 1.0)
    out_val = tl.where(has, acc / l_safe, 0.0)
    lse_val = tl.where(has, m_i + tl.log(l_safe), float("-inf"))
    tl.store(
        OUT + row * (HEADS * 192) + head * 192 + channel,
        out_val.to(tl.bfloat16),
        mask=cmask,
    )
    tl.store(LSE + row * HEADS + head, lse_val)


def launch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    packed_regions: torch.Tensor,
    counts: torch.Tensor,
    causal_lengths: torch.Tensor,
    valid_rows: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    softmax_scale: float,
) -> None:
    """Run one program per (row, head); K/V are native [slots,1,192] BF16.

    Single-pass decode with no split-KV partitioning or merge, so ``out`` is the
    final BF16 attention of shape ``[rows, heads, 192]`` and ``lse`` the FP32
    natural log-sum-exp ``[rows, heads]``. ``heads`` and ``packed_width`` are
    compile-time specialization keys.
    """
    if k.dtype != torch.bfloat16:
        raise ValueError("Step5 Triton sparse decode supports only BF16 K/V")
    rows, heads, _ = q.shape
    packed_width = packed_regions.shape[1]
    max_phys_regions = k.shape[0] // 8
    _sparse_decode[(rows, heads)](
        q,
        k,
        v,
        packed_regions,
        counts,
        causal_lengths,
        valid_rows,
        out,
        lse,
        rows,
        max_phys_regions,
        softmax_scale,
        heads,
        packed_width,
        32,
        num_warps=4,
        enable_fp_fusion=False,
    )
