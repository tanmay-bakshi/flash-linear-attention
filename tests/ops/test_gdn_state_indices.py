"""Indexed recurrent state ownership and equality with contiguous execution."""

import pytest
import torch

from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule


@pytest.mark.parametrize('state_v_first', [False, True])
@pytest.mark.parametrize('varlen', [False, True])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@torch.no_grad()
def test_indexed_state_preserves_unselected_rows(state_v_first: bool, varlen: bool, dtype: torch.dtype) -> None:
    """Indexed and contiguous calls apply the same recurrence to the same original rows.

    :param state_v_first: Whether state stores value channels before key channels.
    :param varlen: Whether sequences use cumulative sequence lengths.
    :param dtype: Projection operand dtype.
    """
    device = torch.device('cuda')
    generator = torch.Generator(device=device).manual_seed(31)
    batch, steps, heads, key_dim, value_dim = 3, 4, 2, 32, 24
    state_shape = (7, heads, value_dim, key_dim) if state_v_first else (7, heads, key_dim, value_dim)
    state = torch.randn(state_shape, device=device, generator=generator)
    saved = state.clone()
    rows = torch.tensor([5, 0, 3], device=device)
    reference = state[rows].clone()
    q, k = [torch.randn(batch, steps, heads, key_dim, device=device, dtype=dtype, generator=generator) for _ in range(2)]
    v = torch.randn(batch, steps, heads, value_dim, device=device, dtype=dtype, generator=generator)
    g = -torch.rand(batch, steps, heads, device=device, generator=generator)
    beta = torch.rand(batch, steps, heads, device=device, generator=generator)
    cu_seqlens = torch.tensor([0, 2, 5, 12], device=device) if varlen else None
    if varlen:
        q, k, v, g, beta = [value.flatten(0, 1).unsqueeze(0) for value in (q, k, v, g, beta)]
    for _ in range(3):
        actual, final = fused_recurrent_gated_delta_rule(
            q, k, v, g, beta=beta, initial_state=state, output_final_state=True,
            inplace_final_state=True, state_indices=rows, state_v_first=state_v_first,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
        )
        expected, _ = fused_recurrent_gated_delta_rule(
            q, k, v, g, beta=beta, initial_state=reference, output_final_state=True,
            inplace_final_state=True, state_v_first=state_v_first,
            use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens,
        )
        assert final.data_ptr() == state.data_ptr()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(state[rows], reference, atol=0, rtol=0)
        torch.testing.assert_close(state[[1, 2, 4, 6]], saved[[1, 2, 4, 6]], atol=0, rtol=0)
