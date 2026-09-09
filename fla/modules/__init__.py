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
    'BitLinear': 'fused_bitlinear',
    'FusedBitLinear': 'fused_bitlinear',
    'FusedCrossEntropyLoss': 'fused_cross_entropy',
    'FusedKLDivLoss': 'fused_kl_div',
    'FusedLayerNormGated': 'fused_norm_gate',
    'FusedLayerNormSwishGate': 'fused_norm_gate',
    'FusedLayerNormSwishGateLinear': 'fused_norm_gate',
    'FusedLinearCrossEntropyLoss': 'fused_linear_cross_entropy',
    'FusedRMSNormGated': 'fused_norm_gate',
    'FusedRMSNormSwishGate': 'fused_norm_gate',
    'FusedRMSNormSwishGateLinear': 'fused_norm_gate',
    'GatedMLP': 'mlp',
    'GroupNorm': 'layernorm',
    'GroupNormLinear': 'layernorm',
    'ImplicitLongConvolution': 'convolution',
    'L2Norm': 'l2norm',
    'LayerNorm': 'layernorm',
    'LayerNormLinear': 'layernorm',
    'LongConvolution': 'convolution',
    'RMSNorm': 'layernorm',
    'RMSNormLinear': 'layernorm',
    'RotaryEmbedding': 'rotary',
    'ShortConvolution': 'convolution',
    'TokenShift': 'token_shift',
}

__all__ = [
    'BitLinear',
    'FusedBitLinear',
    'FusedCrossEntropyLoss',
    'FusedKLDivLoss',
    'FusedLayerNormGated',
    'FusedLayerNormSwishGate',
    'FusedLayerNormSwishGateLinear',
    'FusedLinearCrossEntropyLoss',
    'FusedRMSNormGated',
    'FusedRMSNormSwishGate',
    'FusedRMSNormSwishGateLinear',
    'GatedMLP',
    'GroupNorm',
    'GroupNormLinear',
    'ImplicitLongConvolution',
    'L2Norm',
    'LayerNorm',
    'LayerNormLinear',
    'LongConvolution',
    'RMSNorm',
    'RMSNormLinear',
    'RotaryEmbedding',
    'ShortConvolution',
    'TokenShift',
]


def __getattr__(name: str):
    submodule = _SUBMODULES.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
