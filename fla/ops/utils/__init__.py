# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/fla-org/flash-linear-attention/graphs/contributors

import importlib

# Names resolve to their submodule on first access, so importing one kernel family does not pay for the
# whole library: several families import torch.compile / DTensor at module level, which costs seconds.
_SUBMODULES = {
    'addmm': 'matmul',
    'chunk_global_cumsum': 'cumsum',
    'chunk_global_cumsum_scalar': 'cumsum',
    'chunk_global_cumsum_vector': 'cumsum',
    'chunk_local_cumsum': 'cumsum',
    'chunk_local_cumsum_scalar': 'cumsum',
    'chunk_local_cumsum_vector': 'cumsum',
    'get_max_num_splits': 'index',
    'logsumexp_fwd': 'logsumexp',
    'matmul': 'matmul',
    'mean_pooling': 'pooling',
    'pack_sequence': 'pack',
    'prepare_block_csr': 'csr',
    'prepare_chunk_indices': 'index',
    'prepare_chunk_offsets': 'index',
    'prepare_cu_seqlens_from_lens': 'index',
    'prepare_cu_seqlens_from_mask': 'index',
    'prepare_lens': 'index',
    'prepare_lens_from_mask': 'index',
    'prepare_position_ids': 'index',
    'prepare_sequence_ids': 'index',
    'prepare_token_indices': 'index',
    'softmax_bwd': 'softmax',
    'softmax_fwd': 'softmax',
    'softplus': 'softplus',
    'solve_tril': 'solve_tril',
    'unpack_sequence': 'pack',
}

__all__ = [
    "addmm",
    "chunk_global_cumsum",
    "chunk_global_cumsum_scalar",
    "chunk_global_cumsum_vector",
    "chunk_local_cumsum",
    "chunk_local_cumsum_scalar",
    "chunk_local_cumsum_vector",
    "get_max_num_splits",
    "logsumexp_fwd",
    "matmul",
    "mean_pooling",
    "pack_sequence",
    "prepare_block_csr",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
    "prepare_cu_seqlens_from_lens",
    "prepare_cu_seqlens_from_mask",
    "prepare_lens",
    "prepare_lens_from_mask",
    "prepare_position_ids",
    "prepare_sequence_ids",
    "prepare_token_indices",
    "softmax_bwd",
    "softmax_fwd",
    "softplus",
    "solve_tril",
    "unpack_sequence",
]


def __getattr__(name: str):
    submodule = _SUBMODULES.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
