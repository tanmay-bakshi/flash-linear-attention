"""FP32 chunk-output gradients against independent float64 forward equations."""

import math

import pytest
import torch

from fla.ops.common.chunk_o import chunk_bwd_dqkwg
from fla.utils import device


def _chunk_vjp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    do: torch.Tensor,
    h: torch.Tensor,
    dh: torch.Tensor,
    g: torch.Tensor | None,
    dv: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Differentiate the local output and the state at the end of one chunk.

    :param q: Queries in token, head, key order.
    :param k: Keys in token, head, key order.
    :param v: Values in token, head, value order.
    :param do: Output cotangent.
    :param h: State at the start of the chunk in head, key, value order.
    :param dh: Cotangent of the state at the end of the chunk.
    :param g: Cumulative log2 decay, or ``None`` for an ungated rule.
    :param dv: Value cotangent used to differentiate the WY correction.
    :returns: Query, key, WY correction, and cumulative gate gradients.
    """
    queries = q.double().detach().requires_grad_(True)
    keys = k.double().detach().requires_grad_(True)
    gate = g.double().detach().requires_grad_(True) if g is not None else None
    group = v.shape[1] // q.shape[1]
    qh = queries.repeat_interleave(group, dim=1).transpose(0, 1)
    kh = keys.repeat_interleave(group, dim=1).transpose(0, 1)
    vh = v.double().transpose(0, 1)
    state = h.double()
    causal = torch.ones(q.shape[0], q.shape[0], dtype=torch.bool, device=q.device).tril()
    scores = qh @ kh.transpose(-1, -2)
    if gate is None:
        output = (qh @ state + scores.masked_fill(~causal, 0) @ vh) * q.shape[-1] ** -0.5
        final = state + kh.transpose(-1, -2) @ vh
    else:
        gh = gate.transpose(0, 1)
        differences = (gh[:, :, None] - gh[:, None, :]).masked_fill(~causal, 0)
        decay = differences.exp2().masked_fill(~causal, 0)
        output = ((qh * gh.exp2()[:, :, None]) @ state + (scores * decay) @ vh) * q.shape[-1] ** -0.5
        final = state * gh[:, -1, None, None].exp2() + kh.transpose(-1, -2) @ (vh * (gh[:, -1:] - gh).exp2()[:, :, None])
    loss = (output * do.double().transpose(0, 1)).sum() + (final * dh.double()).sum()
    leaves = (queries, keys) if gate is None else (queries, keys, gate)
    gradients = torch.autograd.grad(loss, leaves)
    dw = -(dv.double().transpose(0, 1) @ state.transpose(-1, -2)).transpose(0, 1) if dv is not None else None
    dg = gradients[2] / math.log(2) if gate is not None else None
    return gradients[0], gradients[1], dw, dg


@pytest.mark.parametrize(
    'lengths,key_dim,value_dim,heads,value_heads,decay,use_w,state_v_first',
    [
        ((619, 619), 128, 128, 16, 16, 0.125, True, False),
        ((1, 63, 64, 65, 129), 128, 128, 2, 4, 0.125, True, False),
        ((1, 63, 64, 65, 129), 128, 128, 2, 4, 0.125, True, True),
        ((65, 127), 96, 80, 2, 4, 0.0, True, False),
        ((65, 127), 96, 80, 2, 4, 2.0, True, True),
        ((129,), 128, 128, 2, 2, None, True, False),
        ((129,), 128, 128, 2, 2, None, False, True),
        ((129,), 128, 128, 2, 2, 0.125, False, False),
    ],
)
def test_chunk_dqkwg_fp32_vjp(
    lengths: tuple[int, ...],
    key_dim: int,
    value_dim: int,
    heads: int,
    value_heads: int,
    decay: float | None,
    use_w: bool,
    state_v_first: bool,
) -> None:
    """Check tails, packed boundaries, grouped heads, state layouts, and optional gradients."""
    torch.manual_seed(104)
    total = sum(lengths)
    chunk_size = 64
    chunk_count = sum((length + chunk_size - 1) // chunk_size for length in lengths)
    q = torch.randn(1, total, heads, key_dim, device=device) * key_dim ** -0.5
    k = torch.randn_like(q) * key_dim ** -0.5
    v = torch.randn(1, total, value_heads, value_dim, device=device)
    do = torch.randn_like(v)
    h = torch.randn(1, chunk_count, value_heads, key_dim, value_dim, device=device) * 0.1
    dh = torch.randn_like(h) * 0.1
    dv = torch.randn_like(v) if use_w else None
    w = torch.empty(1, total, value_heads, key_dim, device=device) if use_w else None
    g = torch.empty(1, total, value_heads, device=device) if decay is not None else None
    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    expected: list[list[torch.Tensor]] = [[], [], [], []]
    chunk_index = 0
    for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        for offset in range(start, end, chunk_size):
            stop = min(offset + chunk_size, end)
            if g is not None and decay is not None:
                g[:, offset:stop] = -(torch.rand_like(g[:, offset:stop]) * decay).cumsum(1)
            gradients = _chunk_vjp(q[0, offset:stop], k[0, offset:stop], v[0, offset:stop], do[0, offset:stop],
                                   h[0, chunk_index], dh[0, chunk_index],
                                   g[0, offset:stop] if g is not None else None,
                                   dv[0, offset:stop] if dv is not None else None)
            for output, gradient in zip(expected, gradients, strict=True):
                if gradient is not None:
                    output.append(gradient)
            chunk_index += 1
    if state_v_first:
        h = h.transpose(-1, -2).contiguous()
        dh = dh.transpose(-1, -2).contiguous()
    actual = chunk_bwd_dqkwg(q=q, k=k, v=v, do=do, h=h, dh=dh, w=w, g=g, dv=dv,
                            scale=key_dim ** -0.5, state_v_first=state_v_first,
                            cu_seqlens=torch.tensor(boundaries, device=device, dtype=torch.int32))
    for name, got, pieces in zip(('dq', 'dk', 'dw', 'dg'), actual, expected, strict=True):
        if len(pieces) == 0:
            assert got is None, name
            continue
        assert got is not None, name
        want = torch.cat(pieces).unsqueeze(0)
        relative = (got.double() - want).norm() / want.norm().clamp_min(1e-12)
        assert relative < 2e-5, (name, relative)
        torch.testing.assert_close(got.double(), want, atol=2e-5, rtol=2e-4)
