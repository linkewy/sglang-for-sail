# SPDX-License-Identifier: Apache-2.0
"""Step5 grouped-union sparse paged-KV prefill in portable Triton (SM90 replacement).

Mirrors ``cute/sm90/prefill_union_paged_sm90_gqa.py`` for devices that cannot run
the CuTe DSL kernel (e.g. sm80 PPU). Grid is ``(group, head)``; each program walks
the up-to-16 query rows of one group sequentially and runs an online softmax over
that group's union of eight-token regions.

A union entry ``window_idx`` expands to eight KV tokens::

    phys_region   = union_physical[group, window_idx]   # physical region id
    logical_region = union_logical[group, window_idx]    # logical region id
    phys_tok      = phys_region * 8 + lane               # -> K/V slot
    logical_tok   = logical_region * 8 + lane            # -> causal position

A key is usable only when all three hold, exactly as the SM90 exact-union mask:

1. physical slot valid: ``0 <= phys_region < slots // 8``
2. membership selects this query: ``(membership[group, window_idx] >> q_slot) & 1``
3. causal: ``logical_region >= 0`` and ``logical_tok <= query_positions[row]``

GQA has a single KV head, so every logical head shares the same K/V. Output is
BF16; LSE is the natural logarithm of the softmax denominator. Empty rows write
zero output and ``-inf`` LSE; rows outside the group's range are left untouched.
"""

import torch
import triton
import triton.language as tl

REGION_TOKENS = tl.constexpr(8)


@triton.jit(
    do_not_specialize=("ROWS", "SLOTS_DIV8"),
    do_not_specialize_on_alignment=("ROWS", "SLOTS_DIV8"),
)
def _sparse_prefill(
    Q,
    K,
    V,
    UNION_PHYS,
    UNION_LOGICAL,
    UNION_COUNTS,
    MEMBERSHIP,
    GSTART,
    GEND,
    QPOS,
    OUT,
    LSE,
    ROWS,
    SLOTS_DIV8,
    SOFTMAX_SCALE,
    HEADS: tl.constexpr,
    CAPACITY: tl.constexpr,
    BLOCK_N: tl.constexpr,
    Q_GROUP: tl.constexpr,
):
    group = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)

    q_base = tl.load(GSTART + group)
    q_end = tl.load(GEND + group)
    q_tile_count = tl.minimum(tl.maximum(q_end - q_base, 0), Q_GROUP)
    union_count = tl.minimum(tl.load(UNION_COUNTS + group), CAPACITY)
    count_tokens = union_count * REGION_TOKENS

    channel = tl.arange(0, 256)
    cmask = channel < 192

    for q_slot in range(q_tile_count):
        row = (q_base + q_slot).to(tl.int64)
        if row < ROWS:
            q_pos = tl.load(QPOS + row)
            q_vec = tl.load(
                Q + row * (HEADS * 192) + head * 192 + channel, mask=cmask, other=0.0
            ).to(tl.float32)

            m_i = float("-inf")
            l_i = 0.0
            acc = tl.zeros((256,), dtype=tl.float32)

            for start in range(0, count_tokens, BLOCK_N):
                tok = start + tl.arange(0, BLOCK_N)
                tvalid = tok < count_tokens
                win = tok // REGION_TOKENS
                lane = tok - win * REGION_TOKENS

                phys_region = tl.load(UNION_PHYS + group * CAPACITY + win, mask=tvalid, other=-1)
                logical_region = tl.load(UNION_LOGICAL + group * CAPACITY + win, mask=tvalid, other=-1)
                memb = tl.load(MEMBERSHIP + group * CAPACITY + win, mask=tvalid, other=0)

                selected = ((memb >> q_slot) & 1) != 0
                valid_phys = (phys_region >= 0) & (phys_region < SLOTS_DIV8)
                phys_tok = (phys_region * REGION_TOKENS + lane).to(tl.int64)
                logical_tok = logical_region * REGION_TOKENS + lane
                keep = (
                    tvalid
                    & valid_phys
                    & selected
                    & (logical_region >= 0)
                    & (logical_tok <= q_pos)
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
    union_physical: torch.Tensor,
    union_logical: torch.Tensor,
    union_counts: torch.Tensor,
    membership: torch.Tensor,
    group_query_start: torch.Tensor,
    group_query_end: torch.Tensor,
    query_positions: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor,
    *,
    softmax_scale: float,
) -> None:
    """Run one program per (group, head); K/V are native [slots,1,192] BF16.

    Argument order mirrors the CuTe ``compile_prefill`` call so
    ``step5_sparse_prefill`` can pick a backend without reshaping its call site.
    ``heads`` and ``capacity`` are compile-time specialization keys, matching the
    CuTe dispatch cache key.
    """
    rows, heads, _ = q.shape
    groups, capacity = union_physical.shape
    slots_div8 = k.shape[0] // 8
    # Max packed query rows per group, matching the CuTe q_group contract.
    q_group = 128 // heads
    _sparse_prefill[(groups, heads)](
        q,
        k,
        v,
        union_physical,
        union_logical,
        union_counts,
        membership,
        group_query_start,
        group_query_end,
        query_positions,
        out,
        lse,
        rows,
        slots_div8,
        softmax_scale,
        heads,
        capacity,
        32,
        q_group,
        num_warps=4,
        enable_fp_fusion=False,
    )
