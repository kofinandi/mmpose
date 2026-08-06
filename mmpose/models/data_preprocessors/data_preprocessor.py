# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseDataPreprocessor, ImgDataPreprocessor
from mmengine.utils import is_seq_of

from mmpose.registry import MODELS


@MODELS.register_module()
class PoseDataPreprocessor(ImgDataPreprocessor):
    """Image pre-processor for pose estimation tasks.

    Comparing with the :class:`ImgDataPreprocessor`,

    1. It will additionally append batch_input_shape
    to data_samples considering the DETR-based pose estimation tasks.

    2. Support image augmentation transforms on batched data.

    It provides the data pre-processing as follows

    - Collate and move data to the target device.
    - Pad inputs to the maximum size of current batch with defined
      ``pad_value``. The padding size can be divisible by a defined
      ``pad_size_divisor``
    - Stack inputs to batch_inputs.
    - Convert inputs from bgr to rgb if the shape of input is (3, H, W).
    - Normalize image with defined std and mean.
    - Apply batch augmentation transforms.

    Args:
        mean (sequence of float, optional): The pixel mean of R, G, B
            channels. Defaults to None.
        std (sequence of float, optional): The pixel standard deviation
            of R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded image should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert image from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert image from RGB to BGR.
            Defaults to False.
        non_blocking (bool): Whether block current process
            when transferring data to device. Defaults to False.
        batch_augments: (list of dict, optional): Configs of augmentation
            transforms on batched data. Defaults to None.
    """

    def __init__(self,
                 mean: Sequence[float] = None,
                 std: Sequence[float] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False,
                 batch_augments: Optional[List[dict]] = None):
        super().__init__(
            mean=mean,
            std=std,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            bgr_to_rgb=bgr_to_rgb,
            rgb_to_bgr=rgb_to_bgr,
            non_blocking=non_blocking)

        if batch_augments is not None:
            self.batch_augments = nn.ModuleList(
                [MODELS.build(aug) for aug in batch_augments])
        else:
            self.batch_augments = None

    def forward(self, data: dict, training: bool = False) -> dict:
        """Perform normalization, padding and bgr2rgb conversion based on
        ``BaseDataPreprocessor``.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.

        Returns:
            dict: Data in the same format as the model input.
        """
        batch_pad_shape = self._get_pad_shape(data)
        data = super().forward(data=data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']

        # update metainfo since the image shape might change
        batch_input_shape = tuple(inputs[0].size()[-2:])
        for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
            data_sample.set_metainfo({
                'batch_input_shape': batch_input_shape,
                'pad_shape': pad_shape
            })

        # apply batch augmentations
        if training and self.batch_augments is not None:
            for batch_aug in self.batch_augments:
                inputs, data_samples = batch_aug(inputs, data_samples)

        return {'inputs': inputs, 'data_samples': data_samples}

    def _get_pad_shape(self, data: dict) -> List[tuple]:
        """Get the pad_shape of each image based on data and
        pad_size_divisor."""
        _batch_inputs = data['inputs']
        # Process data with `pseudo_collate`.
        if is_seq_of(_batch_inputs, torch.Tensor):
            batch_pad_shape = []
            for ori_input in _batch_inputs:
                pad_h = int(
                    np.ceil(ori_input.shape[1] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                pad_w = int(
                    np.ceil(ori_input.shape[2] /
                            self.pad_size_divisor)) * self.pad_size_divisor
                batch_pad_shape.append((pad_h, pad_w))
        # Process data with `default_collate`.
        elif isinstance(_batch_inputs, torch.Tensor):
            assert _batch_inputs.dim() == 4, (
                'The input of `ImgDataPreprocessor` should be a NCHW tensor '
                'or a list of tensor, but got a tensor with shape: '
                f'{_batch_inputs.shape}')
            pad_h = int(
                np.ceil(_batch_inputs.shape[1] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            pad_w = int(
                np.ceil(_batch_inputs.shape[2] /
                        self.pad_size_divisor)) * self.pad_size_divisor
            batch_pad_shape = [(pad_h, pad_w)] * _batch_inputs.shape[0]
        else:
            raise TypeError('Output of `cast_data` should be a dict '
                            'or a tuple with inputs and data_samples, but got'
                            f'{type(data)}: {data}')
        return batch_pad_shape


@MODELS.register_module()
class ClipPoseDataPreprocessor(BaseDataPreprocessor):
    """Data pre-processor for video (multi-frame clip) topdown/bottomup pose
    models.

    ``PoseDataPreprocessor`` (via mmengine's ``ImgDataPreprocessor``) assumes
    every per-sample input tensor is ``(C, H, W)``.  Clip-based models
    (Poseidon, TAR-ViTPose, PAVE-Net, ...) instead receive a per-sample
    ``(T, C, H, W)`` tensor -- ``T`` input frames sharing the same affine crop
    (topdown) or the same padded canvas (bottomup) -- which after batching is
    ``(B, T, C, H, W)``. This class re-implements the normalisation / BGR2RGB
    / stacking steps of ``PoseDataPreprocessor`` for that extra leading frame
    dimension, and otherwise keeps the same interface (``mean``, ``std``,
    ``bgr_to_rgb``, ``pad_size_divisor``, ...) so clip models can be swapped
    in without touching the rest of the config.

    Padding across a batch is only supported when every sample already has
    the same spatial size (always true for the topdown affine-cropped clips
    produced by this benchmark), in which case ``pad_size_divisor`` simply
    rounds the shared size up; mismatched sizes raise an error rather than
    silently padding per-frame, since doing so would shift the per-frame
    affine crops relative to each other.

    Args:
        mean (sequence of float, optional): The pixel mean of R, G, B
            channels. Defaults to None.
        std (sequence of float, optional): The pixel standard deviation
            of R, G, B channels. Defaults to None.
        pad_size_divisor (int): The size of padded frames should be
            divisible by ``pad_size_divisor``. Defaults to 1.
        pad_value (float or int): The padded pixel value. Defaults to 0.
        bgr_to_rgb (bool): whether to convert frames from BGR to RGB.
            Defaults to False.
        rgb_to_bgr (bool): whether to convert frames from RGB to BGR.
            Defaults to False.
        non_blocking (bool): Whether block current process
            when transferring data to device. Defaults to False.
    """

    def __init__(self,
                 mean: Optional[Sequence[float]] = None,
                 std: Optional[Sequence[float]] = None,
                 pad_size_divisor: int = 1,
                 pad_value: Union[float, int] = 0,
                 bgr_to_rgb: bool = False,
                 rgb_to_bgr: bool = False,
                 non_blocking: Optional[bool] = False):
        super().__init__(non_blocking=non_blocking)
        assert not (bgr_to_rgb and rgb_to_bgr), (
            '`bgr_to_rgb` and `rgb_to_bgr` cannot both be True')
        assert (mean is None) == (std is None), (
            'mean and std should be both None or both provided')
        if mean is not None:
            assert len(mean) in (1, 3) and len(std) in (1, 3), (
                '`mean`/`std` should have 1 or 3 values, but got '
                f'{len(mean)}/{len(std)}')
            self._enable_normalize = True
            # (B, T, C, H, W): mean/std broadcast over the channel dim (2).
            self.register_buffer('mean',
                                 torch.tensor(mean).view(1, 1, -1, 1, 1),
                                 False)
            self.register_buffer('std',
                                 torch.tensor(std).view(1, 1, -1, 1, 1),
                                 False)
        else:
            self._enable_normalize = False
        self._channel_conversion = bgr_to_rgb or rgb_to_bgr
        self.pad_size_divisor = pad_size_divisor
        self.pad_value = pad_value

    def forward(self, data: dict, training: bool = False) -> dict:
        """Normalise, optionally BGR<->RGB-convert, and stack a batch of
        ``(T, C, H, W)`` clip tensors into ``(B, T, C, H, W)``.

        Args:
            data (dict): Data sampled from the dataloader / benchmark
                harness. ``data['inputs']`` is either a list of per-sample
                ``(T, C, H, W)`` tensors (``pseudo_collate``) or an already
                stacked ``(B, T, C, H, W)`` tensor (``default_collate``).
            training (bool): Unused; kept for interface parity with
                ``PoseDataPreprocessor``.

        Returns:
            dict: ``{'inputs': Tensor[B, T, C, H, W], 'data_samples': ...}``
        """
        data = self.cast_data(data)
        batch_inputs = data['inputs']

        if is_seq_of(batch_inputs, torch.Tensor):
            shapes = {tuple(t.shape) for t in batch_inputs}
            if len(shapes) > 1:
                raise ValueError(
                    'ClipPoseDataPreprocessor requires all samples in a '
                    f'batch to share the same (T, C, H, W) shape, got '
                    f'{shapes}. This is expected for topdown affine-cropped '
                    'clips; check the pipeline / batch construction.')
            batch_inputs = torch.stack(list(batch_inputs), dim=0)
        elif not isinstance(batch_inputs, torch.Tensor):
            raise TypeError(
                'Output of `cast_data` should be a dict with `inputs` as a '
                'Tensor or list of Tensors, but got '
                f'{type(batch_inputs)}: {batch_inputs}')

        assert batch_inputs.dim() == 5, (
            'ClipPoseDataPreprocessor expects a (B, T, C, H, W) input, but '
            f'got a tensor with shape: {tuple(batch_inputs.shape)}')

        if self._channel_conversion:
            batch_inputs = batch_inputs[:, :, [2, 1, 0], ...]

        batch_inputs = batch_inputs.float()

        if self._enable_normalize:
            batch_inputs = (batch_inputs - self.mean) / self.std

        h, w = batch_inputs.shape[-2:]
        target_h = int(np.ceil(h / self.pad_size_divisor)) * \
            self.pad_size_divisor
        target_w = int(np.ceil(w / self.pad_size_divisor)) * \
            self.pad_size_divisor
        pad_h, pad_w = target_h - h, target_w - w
        if pad_h or pad_w:
            batch_inputs = nn.functional.pad(
                batch_inputs, (0, pad_w, 0, pad_h), 'constant',
                self.pad_value)

        data_samples = data.get('data_samples')
        batch_input_shape = tuple(batch_inputs.shape[-2:])
        if data_samples is not None:
            for data_sample in data_samples:
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': batch_input_shape,
                })

        return {'inputs': batch_inputs, 'data_samples': data_samples}
