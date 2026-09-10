"""FP32 WY values and vector-Jacobian products against float64 triangular solves."""

import math

import pytest
import torch

from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd


def _solve(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None,
    chunk_size: int,
    lengths: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate each independent WY system using differentiable float64 solves.

    :param k: Keys in batch, time, key-head, channel order.
    :param v: Values in batch, time, value-head, channel order.
    :param beta: Token write strengths.
    :param g: Cumulative natural-log decay within each chunk, or no decay.
    :param chunk_size: Number of tokens in a full chunk.
    :param lengths: Independent sequence lengths inside each batch row.
    :returns: Transformed keys and values.
    """
    group = v.shape[2] // k.shape[2]
    keys = k.repeat_interleave(group, dim=2).transpose(1, 2)
    values = v.transpose(1, 2)
    strength = beta.transpose(1, 2)
    decay = g.transpose(1, 2) if g is not None else torch.zeros_like(strength)
    ws: list[torch.Tensor] = []
    us: list[torch.Tensor] = []
    start = 0
    for length in lengths:
        for offset in range(0, length, chunk_size):
            begin, end = start + offset, start + min(offset + chunk_size, length)
            kc, vc, bc, gc = keys[:, :, begin:end], values[:, :, begin:end], strength[:, :, begin:end], decay[:, :, begin:end]
            lower = ((kc * bc[..., None]) @ kc.transpose(-1, -2) * (gc[..., :, None] - gc[..., None, :]).exp()).tril(-1)
            matrix = lower + torch.eye(end - begin, dtype=k.dtype, device=k.device)
            rhs = torch.cat((kc * (bc * gc.exp())[..., None], vc * bc[..., None]), dim=-1)
            solved = torch.linalg.solve_triangular(matrix, rhs, upper=False, unitriangular=True)
            ws.append(solved[..., :k.shape[-1]])
            us.append(solved[..., k.shape[-1]:])
        start += length
    return torch.cat(ws, dim=2).transpose(1, 2), torch.cat(us, dim=2).transpose(1, 2)


def _assert_fp32(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Check elementwise and whole-tensor error without hiding small gradients.

    :param actual: Kernel output.
    :param expected: Independent double-precision result.
    """
    expected = expected.detach()
    relative = float((actual.double() - expected).norm() / expected.norm().clamp_min(1e-12))
    assert relative < 2e-5, relative
    torch.testing.assert_close(actual.double(), expected, atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize(
    ('batch', 'lengths', 'heads', 'value_heads', 'key_dim', 'value_dim', 'chunk_size', 'use_g'),
    [
        (2, (619,), 16, 16, 128, 128, 64, True),
        (2, (127,), 2, 4, 48, 80, 64, True),
        (1, (65, 3, 130), 2, 4, 128, 96, 64, True),
        (1, (17, 65), 2, 2, 32, 48, 32, True),
        (2, (129,), 2, 2, 128, 128, 64, False),
        (1, (31, 66), 2, 4, 48, 80, 16, False),
    ],
)
def test_wy_values_and_vjps_match_float64(
    batch: int,
    lengths: tuple[int, ...],
    heads: int,
    value_heads: int,
    key_dim: int,
    value_dim: int,
    chunk_size: int,
    use_g: bool,
) -> None:
    """Validate gated, ungated, grouped-head, packed, and partial-tile systems.

    :param batch: Number of batch rows.
    :param lengths: Sequence lengths within each row.
    :param heads: Key-head count.
    :param value_heads: Value-head count.
    :param key_dim: Key channels.
    :param value_dim: Value channels.
    :param chunk_size: WY block size.
    :param use_g: Whether to include gate derivatives.
    """
    torch.manual_seed(173)
    seq = sum(lengths)
    k = torch.nn.functional.normalize(torch.randn(batch, seq, heads, key_dim, device='cuda', dtype=torch.float32), dim=-1)
    v = torch.randn(batch, seq, value_heads, value_dim, device='cuda', dtype=torch.float32)
    beta = torch.rand(batch, seq, value_heads, device='cuda', dtype=torch.float32)
    g = -.25 * torch.rand_like(beta) if use_g else None
    start = 0
    for length in lengths:
        for offset in range(0, length, chunk_size):
            begin, end = start + offset, start + min(offset + chunk_size, length)
            if g is not None:
                g[:, begin:end] = g[:, begin:end].cumsum(dim=1)
        start += length
    cu_seqlens = None
    if len(lengths) > 1:
        cu_seqlens = torch.tensor((0, *lengths), device='cuda', dtype=torch.int64).cumsum(0)
    inputs = [k, v, beta] if g is None else [k, v, beta, g]
    leaves = [value.double().requires_grad_(True) for value in inputs]
    expected_w, expected_u = _solve(*leaves[:3], leaves[3] if g is not None else None, chunk_size, lengths)
    g_log2 = g * math.log2(math.e) if g is not None else None
    w, u, inverse = chunk_gated_delta_rule_fwd_intra(k=k, v=v, beta=beta, g=g_log2, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    dw, du = torch.randn_like(w), torch.randn_like(u)
    expected_grads = torch.autograd.grad((expected_w * dw.double()).sum() + (expected_u * du.double()).sum(), leaves)
    gradients = prepare_wy_repr_bwd(k=k, v=v, beta=beta, A=inverse, dw=dw, du=du, g=g_log2, cu_seqlens=cu_seqlens)
    for actual, expected in zip((w, u, *gradients[:len(inputs)]), (expected_w, expected_u, *expected_grads), strict=True):
        _assert_fp32(actual, expected)
    if g is None:
        assert gradients[3] is None
