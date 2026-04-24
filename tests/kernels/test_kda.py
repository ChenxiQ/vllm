# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision tests for vllm's chunk_kda Triton operator on NPU.

Compares the Triton kernel output against a naive PyTorch recurrent reference
implementation. The naive reference computes in float32 internally for numerical
stability, while the Triton kernel operates in fp16/bf16. On NPU, fp16
accumulation error is higher than on GPU, so the tolerance is relaxed
accordingly (ratio=0.10 vs FLA's 0.005 on GPU).
"""

import pytest
import torch
import torch.nn.functional as F

import torch_npu  # noqa: F401

from vllm.model_executor.layers.fla.ops.kda import chunk_kda
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd

DEVICE = "npu"

# NPU triton-ascend produces ~7-11% rmse ratio vs naive recurrent even in fp32
# (same algorithm matches within 0.5% on GPU). This is a triton-ascend compiler
# issue, not dtype precision. FLA uses ratio=0.005 on GPU.
# Both ref and tri use the same Triton l2norm_fwd for normalization, so
# the error is purely from the chunk algorithm on triton-ascend.
# o: actual max ~0.108, set 0.12
# ht: actual max ~0.0025 (fp16/bf16), ~1e-6 (fp32), set 0.003
NPU_RMSE_RATIO_O = 0.005
# NPU_RMSE_RATIO_O = 0.12
NPU_RMSE_RATIO_HT = 0.005


def naive_recurrent_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Naive recurrent KDA reference (pure PyTorch, runs on any device).

    Ported from flash-linear-attention/fla/ops/kda/naive.py.
    No einops dependency.
    """
    dtype = v.dtype
    B, T, H, K, V = *q.shape, v.shape[-1]
    if scale is None:
        scale = K ** -0.5

    q, k, v, g, beta = map(lambda x: x.to(torch.float), [q, k, v, g, beta])
    q = q * scale

    S = k.new_zeros(B, H, K, V).to(q)
    if initial_state is not None:
        S += initial_state
    o = torch.zeros_like(v)
    for i in range(T):
        q_i, k_i, v_i, g_i, b_i = q[:, i], k[:, i], v[:, i], g[:, i], beta[:, i]
        S = S * g_i[..., None].exp()
        S = S + torch.einsum(
            "bhk,bhv->bhkv",
            b_i[..., None] * k_i,
            v_i - (k_i[..., None] * S).sum(-2),
        )
        o[:, i] = torch.einsum("bhk,bhkv->bhv", q_i, S)
    if not output_final_state:
        S = None
    return o.to(dtype), S


def assert_close(name: str, ref: torch.Tensor, tri: torch.Tensor, ratio: float, err_atol: float = 1e-6):
    """RMSE-based relative error comparison (same logic as FLA's assert_close)."""
    abs_err = (ref.detach() - tri.detach()).flatten().abs().max().item()
    rmse_diff = (ref.detach() - tri.detach()).flatten().square().mean().sqrt().item()
    rmse_base = ref.detach().flatten().square().mean().sqrt().item()
    rel_err = rmse_diff / (rmse_base + 1e-8)
    print(f"{name:>4} | max abs err: {abs_err:.6f} | rmse ratio: {rel_err:.6f} | threshold: {ratio}")
    if abs_err <= err_atol:
        return
    assert not torch.isnan(ref).any(), f"{name}: NaN detected in ref"
    assert not torch.isnan(tri).any(), f"{name}: NaN detected in tri"
    assert rel_err < ratio, (
        f"{name}: max abs err {abs_err:.6f}, rmse ratio {rel_err:.6f} >= {ratio}"
    )


@pytest.mark.parametrize(
    ("H", "D", "cu_seqlens", "dtype"),
    [
        pytest.param(
            *test,
            id="H{}-D{}-cu{}-{}".format(*test),
        )
        for test in [
            # varlen (B=1, with cu_seqlens) — matches vllm KimiDeltaAttention usage
            # All cases use use_qk_l2norm_in_kernel=True (vllm always enables it)
            (32, 128, [0, 64], torch.float16),
            (32, 128, [0, 1024], torch.float16),
            (32, 128, [0, 15], torch.float16),
            (32, 128, [0, 256, 512, 768, 1024], torch.float16),
            (32, 128, [0, 15, 100, 300, 1200], torch.float16),
            (64, 128, [0, 256, 500, 1000], torch.float16),
            (32, 128, [0, 8192], torch.float16),
            (32, 128, [0, 256, 500, 1000], torch.bfloat16),
        ]
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_chunk_kda(
    H: int,
    D: int,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):
    T = cu_seqlens[-1]
    if T > 4096:
        pytest.skip("T>4096 hits triton-ascend grid size limit (65536)")

    torch.manual_seed(42)
    B = 1  # vllm always uses B=1 with cu_seqlens
    cu_seqlens_t = torch.LongTensor(cu_seqlens).to(DEVICE)
    N = len(cu_seqlens) - 1

    # Generate inputs on NPU in target dtype
    q = torch.randn(B, T, H, D, dtype=dtype, device=DEVICE)
    k = torch.randn(B, T, H, D, dtype=dtype, device=DEVICE)
    v = torch.randn(B, T, H, D, dtype=dtype, device=DEVICE)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE)).to(dtype)
    beta = torch.rand(B, T, H, dtype=dtype, device=DEVICE).sigmoid()
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=DEVICE)

    # Naive reference on NPU — computes in float32 internally
    # Use l2norm_fwd (same Triton kernel as chunk_kda) so ref and tri share
    # the same normalization path, isolating only the chunk algorithm error.
    ref_outputs = []
    ref_states = []
    for i in range(N):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        q_i = l2norm_fwd(q[:, s:e].contiguous())
        k_i = l2norm_fwd(k[:, s:e].contiguous())
        o_i, ht_i = naive_recurrent_kda(
            q_i,
            k_i,
            v[:, s:e],
            g[:, s:e],
            beta[:, s:e],
            initial_state=h0[i],
            output_final_state=True,
        )
        ref_outputs.append(o_i)
        ref_states.append(ht_i)
    ref_o = torch.cat(ref_outputs, dim=1)
    ref_ht = torch.cat(ref_states, dim=0)

    # Triton kernel — same call pattern as KimiDeltaAttention
    tri_o, tri_ht = chunk_kda(
        q=q.clone(),
        k=k.clone(),
        v=v.clone(),
        g=g.clone(),
        beta=beta.clone(),
        initial_state=h0.clone(),
        output_final_state=True,
        cu_seqlens=cu_seqlens_t,
        use_qk_l2norm_in_kernel=True,
    )

    # Verify Triton kernel produces valid output (no crash, no NaN)
    assert not torch.isnan(tri_o).any(), "Triton output o contains NaN"
    assert not torch.isnan(tri_ht).any(), "Triton output ht contains NaN"

    # Precision comparison — naive is float32 ground truth, Triton is fp16/bf16 on NPU
    assert_close("o", ref_o, tri_o, NPU_RMSE_RATIO_O)
    # tri_ht layout is (N, H, V, K); transpose to (N, H, K, V) to match naive
    assert_close("ht", ref_ht, tri_ht.transpose(-1, -2).contiguous(), NPU_RMSE_RATIO_HT)


@pytest.mark.parametrize(
    ("H", "D", "cu_seqlens"),
    [
        pytest.param(
            *test,
            id="H{}-D{}-cu{}".format(*test),
        )
        for test in [
            (32, 128, [0, 64]),
            (32, 128, [0, 1024]),
            (32, 128, [0, 256, 512, 768, 1024]),
            (64, 128, [0, 256, 500, 1000]),
        ]
    ],
)
@pytest.mark.skip_global_cleanup
@torch.inference_mode()
def test_chunk_kda_fp32(
    H: int,
    D: int,
    cu_seqlens: list[int],
):
    """Test chunk_kda with float32 inputs on NPU to isolate algorithmic error."""
    T = cu_seqlens[-1]
    if T > 4096:
        pytest.skip("T>4096 hits triton-ascend grid size limit (65536)")

    torch.manual_seed(42)
    B = 1
    cu_seqlens_t = torch.LongTensor(cu_seqlens).to(DEVICE)
    N = len(cu_seqlens) - 1

    # Generate inputs in float32
    q = torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE)
    k = torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE)
    v = torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE)
    g = F.logsigmoid(torch.randn(B, T, H, D, dtype=torch.float32, device=DEVICE))
    beta = torch.rand(B, T, H, dtype=torch.float32, device=DEVICE).sigmoid()
    h0 = torch.randn(N, H, D, D, dtype=torch.float32, device=DEVICE)

    # Naive reference — also float32
    # Use l2norm_fwd (same Triton kernel as chunk_kda) so ref and tri share
    # the same normalization path, isolating only the chunk algorithm error.
    ref_outputs = []
    ref_states = []
    for i in range(N):
        s, e = cu_seqlens[i], cu_seqlens[i + 1]
        q_i = l2norm_fwd(q[:, s:e].contiguous())
        k_i = l2norm_fwd(k[:, s:e].contiguous())
        o_i, ht_i = naive_recurrent_kda(
            q_i, k_i, v[:, s:e], g[:, s:e], beta[:, s:e],
            initial_state=h0[i], output_final_state=True,
        )
        ref_outputs.append(o_i)
        ref_states.append(ht_i)
    ref_o = torch.cat(ref_outputs, dim=1)
    ref_ht = torch.cat(ref_states, dim=0)

    # Triton kernel in float32
    tri_o, tri_ht = chunk_kda(
        q=q.clone(), k=k.clone(), v=v.clone(), g=g.clone(), beta=beta.clone(),
        initial_state=h0.clone(), output_final_state=True,
        cu_seqlens=cu_seqlens_t, use_qk_l2norm_in_kernel=True,
    )

    assert not torch.isnan(tri_o).any(), "Triton output o contains NaN"
    assert not torch.isnan(tri_ht).any(), "Triton output ht contains NaN"

    # FP32 on NPU still has ~7-11% rmse ratio due to triton-ascend compiler
    assert_close("o", ref_o, tri_o, NPU_RMSE_RATIO_O)
    assert_close("ht", ref_ht, tri_ht.transpose(-1, -2).contiguous(), NPU_RMSE_RATIO_HT)