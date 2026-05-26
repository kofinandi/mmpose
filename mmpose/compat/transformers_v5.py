# Copyright (c) OpenMMLab. All rights reserved.
"""Compatibility shims for mmpretrain with HuggingFace transformers v5.

mmpretrain multimodal modules (BLIP, OFA, etc.) import symbols from
``transformers.modeling_utils`` that were moved or removed in v5.
RF-DETR requires transformers>=5.1.0.  Call ``install_transformers_v5_shims()``
before importing mmdet or mmpretrain when both stacks share an environment.
"""

from typing import Set, Tuple

import torch

_shims_installed = False


def _find_pruneable_heads_and_indices(
    heads,
    n_heads: int,
    head_size: int,
    already_pruned_heads: Set[int],
) -> Tuple[Set[int], torch.LongTensor]:
    """Copied from transformers v4.49 pytorch_utils (removed from public v5 API)."""
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads
    for head in heads:
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()
    return heads, index


def install_transformers_v5_shims() -> None:
    """Re-export transformers v4 symbols onto modeling_utils for mmpretrain."""
    global _shims_installed
    if _shims_installed:
        return

    try:
        import transformers
    except ImportError:
        return

    from mmengine.utils import digit_version

    if digit_version(transformers.__version__) < digit_version('5.0.0'):
        return

    import transformers.modeling_utils as modeling_utils
    from transformers.configuration_utils import PretrainedConfig
    from transformers.generation import GenerationMixin
    from transformers.pytorch_utils import (
        apply_chunking_to_forward,
        prune_linear_layer,
    )

    patches = {
        'apply_chunking_to_forward': apply_chunking_to_forward,
        'prune_linear_layer': prune_linear_layer,
        'find_pruneable_heads_and_indices': _find_pruneable_heads_and_indices,
        'GenerationMixin': GenerationMixin,
        'PretrainedConfig': PretrainedConfig,
    }
    for name, value in patches.items():
        if not hasattr(modeling_utils, name):
            setattr(modeling_utils, name, value)

    _shims_installed = True
