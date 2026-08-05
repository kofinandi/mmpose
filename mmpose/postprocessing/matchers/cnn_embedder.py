# Copyright (c) OpenMMLab. All rights reserved.
"""Detect-and-Track's CNN appearance features over detection crops.

    Girdhar et al., "Detect-and-Track: Efficient Pose Estimation in Videos",
    CVPR 2018.  https://github.com/facebookresearch/DetectAndTrack

Port of ``external/DetectAndTrack/lib/utils/pytorch_cnn_features.py``
(``prepare_image`` / ``extract_features``) together with the cropping done
by ``_compute_deep_features`` in ``lib/core/tracking_engine.py``.

Unlike the other learned components in this package, nothing here is
missing upstream: the "appearance model" is a stock ImageNet-pretrained
torchvision network, so this reproduces the published feature exactly.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
import torch

from ..registry import POST_PROCESS_APPEARANCE_EMBEDDERS
from .base import BaseAppearanceEmbedder

#: Upstream's normalisation constants.  Note the third std entry: upstream
#: writes ``[0.229, 0.224, 0.224]`` where the standard ImageNet value is
#: ``[0.229, 0.224, 0.225]``.  Reproduced by default, see
#: ``keep_upstream_std``.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_UPSTREAM_STD = (0.229, 0.224, 0.224)
_IMAGENET_STD = (0.229, 0.224, 0.225)


@POST_PROCESS_APPEARANCE_EMBEDDERS.register_module()
class TorchvisionCNNEmbedder(BaseAppearanceEmbedder):
    """ImageNet CNN features of each detection's image crop.

    Each box is cropped from the frame with truncated-integer coordinates,
    converted to RGB, resized to 224x224, normalised, and pushed through a
    torchvision classifier up to a named layer; the flattened activation is
    the descriptor.  Distances are cosine, as upstream.

    Upstream ran one forward pass per box; this batches all boxes of a frame
    into a single pass, which is numerically equivalent (the network has no
    cross-sample state and runs in eval mode) and much faster.

    Args:
        model: torchvision classifier name, e.g. ``'resnet18'`` (upstream's
            ``default_model``).
        layer: Name of the child module whose output is the descriptor.
            Upstream default is ``cfg.TRACKING.CNN_MATCHING_LAYER =
            'layer3'``.
        device: Torch device for the network.
        weights: torchvision weights enum name, or ``'DEFAULT'`` for the
            current ImageNet weights (upstream used ``pretrained=True``,
            which is exactly ``'DEFAULT'`` in modern torchvision).
        keep_upstream_std: Reproduce upstream's ``[0.229, 0.224, 0.224]``
            normalisation std, including its apparent typo in the third
            channel.  Set ``False`` for the standard ImageNet
            ``[0.229, 0.224, 0.225]``.  Kept ``True`` by default so the
            shipped feature matches the published one.
        batch_size: Maximum crops per forward pass.
    """

    def __init__(
        self,
        model: str = 'resnet18',
        layer: str = 'layer3',
        device: str = 'cuda:0',
        weights: str = 'DEFAULT',
        keep_upstream_std: bool = True,
        batch_size: int = 64,
    ) -> None:
        from torchvision import models

        if not hasattr(models, model):
            raise ValueError(
                f'Unknown torchvision model {model!r}. Expected e.g. '
                f"'resnet18' (Detect-and-Track's default).")

        if device.startswith('cuda') and not torch.cuda.is_available():
            device = 'cpu'
        self.device = torch.device(device)

        net = getattr(models, model)(weights=weights)
        net.eval().to(self.device)
        if layer not in dict(net.named_children()):
            raise ValueError(
                f'Layer {layer!r} not found in {model!r}; available: '
                f'{list(dict(net.named_children()))}')
        self.model = net
        self.layer = layer
        self.batch_size = int(batch_size)

        std = _UPSTREAM_STD if keep_upstream_std else _IMAGENET_STD
        self._mean = np.array(_IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.array(std, dtype=np.float32).reshape(3, 1, 1)

    def _prepare_crop(self, patch: np.ndarray) -> np.ndarray:
        """Port of upstream ``prepare_image`` for a single crop.

        A crop that OpenCV cannot resize (zero width or height) becomes a
        zero image, exactly as upstream's ``except cv2.error`` branch.
        """
        try:
            patch = cv2.resize(patch[..., (2, 1, 0)], (224, 224))
            patch = patch.transpose(2, 0, 1).astype(np.float32) / 255.0
        except cv2.error:
            patch = np.zeros((3, 224, 224), dtype=np.float32)
        return (patch - self._mean) / self._std

    def _forward_to_layer(self, batch: torch.Tensor) -> torch.Tensor:
        """Run ``batch`` through the network up to :attr:`layer`."""
        x = batch
        for name, module in self.model.named_children():
            x = module(x)
            if name == self.layer:
                return x
        raise RuntimeError(f'Layer {self.layer!r} was never reached.')

    @torch.no_grad()
    def embed(self, image: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        bboxes = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)
        if len(bboxes) == 0:
            return np.zeros((0, 1), dtype=np.float32)
        if image is None:
            raise ValueError(
                'TorchvisionCNNEmbedder.embed() needs frame pixels; the '
                'caller passed image=None.')

        crops: Sequence[np.ndarray] = [
            self._prepare_crop(
                image[int(b[1]):int(b[3]), int(b[0]):int(b[2]), :])
            for b in bboxes
        ]

        feats = []
        for start in range(0, len(crops), self.batch_size):
            chunk = np.stack(crops[start:start + self.batch_size])
            batch = torch.from_numpy(chunk).to(self.device)
            out = self._forward_to_layer(batch)
            feats.append(out.reshape(out.shape[0], -1).cpu().numpy())
        return np.concatenate(feats, axis=0).astype(np.float32)
