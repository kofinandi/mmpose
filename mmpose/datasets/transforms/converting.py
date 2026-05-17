# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple, Union

import numpy as np
from mmcv.transforms import BaseTransform

from mmpose.registry import TRANSFORMS


@TRANSFORMS.register_module()
class KeypointConverter(BaseTransform):
    """Change the order of keypoints according to the given mapping.

    Required Keys:

        - keypoints
        - keypoints_visible

    Modified Keys:

        - keypoints
        - keypoints_visible

    Args:
        num_keypoints (int): The number of keypoints in target dataset.
            Required when ``mapping`` is provided explicitly; inferred
            automatically when ``src`` and ``dst`` are used instead.
        mapping (list): A list containing mapping indexes. Each element has
            format (source_index, target_index).  Mutually exclusive with
            ``src``/``dst``.
        src (str): Source dataset convention name (key in
            :data:`mmpose.datasets.transforms.keypoint_registry.KEYPOINTS`).
            When provided together with ``dst``, ``num_keypoints`` and
            ``mapping`` are derived automatically by matching keypoint names.
        dst (str): Destination dataset convention name.  Must be provided
            together with ``src``.

    Example:
        >>> import numpy as np
        >>> # case 1: 1-to-1 mapping (explicit)
        >>> # (0, 0) means target[0] = source[0]
        >>> self = KeypointConverter(
        >>>     num_keypoints=3,
        >>>     mapping=[
        >>>         (0, 0), (1, 1), (2, 2), (3, 3)
        >>>     ])
        >>> results = dict(
        >>>     keypoints=np.arange(34).reshape(2, 3, 2),
        >>>     keypoints_visible=np.arange(34).reshape(2, 3, 2) % 2)
        >>> results = self(results)
        >>> assert np.equal(results['keypoints'],
        >>>                 np.arange(34).reshape(2, 3, 2)).all()
        >>> assert np.equal(results['keypoints_visible'],
        >>>                 np.arange(34).reshape(2, 3, 2) % 2).all()
        >>>
        >>> # case 2: 2-to-1 mapping (explicit)
        >>> # ((1, 2), 0) means target[0] = (source[1] + source[2]) / 2
        >>> self = KeypointConverter(
        >>>     num_keypoints=3,
        >>>     mapping=[
        >>>         ((1, 2), 0), (1, 1), (2, 2)
        >>>     ])
        >>> results = dict(
        >>>     keypoints=np.arange(34).reshape(2, 3, 2),
        >>>     keypoints_visible=np.arange(34).reshape(2, 3, 2) % 2)
        >>> results = self(results)
        >>>
        >>> # case 3: auto-mapping from dataset convention names
        >>> self = KeypointConverter(src='mpii', dst='coco')
    """

    def __init__(self,
                 num_keypoints: int = None,
                 mapping: Union[List[Tuple[int, int]], List[Tuple[Tuple,
                                                                  int]]] = None,
                 src: str = None,
                 dst: str = None):
        if src is not None and dst is not None:
            from .keypoint_registry import (get_flip_indices, get_keypoints,
                                            get_mapping as _get_mapping)
            if num_keypoints is not None or mapping is not None:
                raise ValueError(
                    'KeypointConverter: provide either (src, dst) for '
                    'automatic mapping or (num_keypoints, mapping) for '
                    'explicit mapping, not both.')
            dst_names = get_keypoints(dst) if isinstance(dst, str) else dst
            num_keypoints = len(dst_names)
            mapping = _get_mapping(src, dst)
            self.dst_flip_indices = get_flip_indices(dst) \
                if isinstance(dst, str) else None
        elif src is not None or dst is not None:
            raise ValueError(
                'KeypointConverter: src and dst must be provided together.')
        else:
            self.dst_flip_indices = None
        if num_keypoints is None or mapping is None:
            raise ValueError(
                'KeypointConverter: either (src, dst) or '
                '(num_keypoints, mapping) must be provided.')

        self.src = src
        self.dst = dst
        self.num_keypoints = num_keypoints
        self.mapping = mapping
        if len(mapping):
            source_index, target_index = zip(*mapping)
        else:
            source_index, target_index = [], []

        src1, src2 = [], []
        interpolation = False
        for x in source_index:
            if isinstance(x, (list, tuple)):
                assert len(x) == 2, 'source_index should be a list/tuple of ' \
                                    'length 2'
                src1.append(x[0])
                src2.append(x[1])
                interpolation = True
            else:
                src1.append(x)
                src2.append(x)

        # When paired source_indexes are input,
        # keep a self.source_index2 for interpolation
        if interpolation:
            self.source_index2 = src2

        self.source_index = src1
        self.target_index = list(target_index)
        self.interpolation = interpolation

    def transform(self, results: dict) -> dict:
        """Transforms the keypoint results to match the target keypoints."""
        num_instances = results['keypoints'].shape[0]

        if 'keypoints_visible' not in results:
            results['keypoints_visible'] = np.ones(
                (num_instances, results['keypoints'].shape[1]))

        if len(results['keypoints_visible'].shape) > 2:
            results['keypoints_visible'] = results['keypoints_visible'][:, :,
                                                                        0]

        # Initialize output arrays
        keypoints = np.zeros((num_instances, self.num_keypoints, 3))
        keypoints_visible = np.zeros((num_instances, self.num_keypoints))
        key = 'keypoints_3d' \
            if results.get('keypoints_3d') is not None else 'keypoints'
        c = results[key].shape[-1]

        flip_indices = results.get('flip_indices', None)

        # Create a mask to weight visibility loss
        keypoints_visible_weights = keypoints_visible.copy()
        keypoints_visible_weights[:, self.target_index] = 1.0

        # Interpolate keypoints if pairs of source indexes provided
        if self.interpolation:
            keypoints[:, self.target_index, :c] = 0.5 * (
                results[key][:, self.source_index] +
                results[key][:, self.source_index2])
            keypoints_visible[:, self.target_index] = results[
                'keypoints_visible'][:, self.source_index] * results[
                    'keypoints_visible'][:, self.source_index2]
            # Flip keypoints if flip_indices provided
            if flip_indices is not None:
                for i, (x1, x2) in enumerate(
                        zip(self.source_index, self.source_index2)):
                    idx = flip_indices[x1] if x1 == x2 else i
                    flip_indices[i] = idx if idx < self.num_keypoints else i
                flip_indices = flip_indices[:len(self.source_index)]
        # Otherwise just copy from the source index
        else:
            keypoints[:,
                      self.target_index, :c] = results[key][:,
                                                            self.source_index]
            keypoints_visible[:, self.target_index] = results[
                'keypoints_visible'][:, self.source_index]

        # Update the results dict
        results['keypoints'] = keypoints[..., :2]
        results['keypoints_visible'] = np.stack(
            [keypoints_visible, keypoints_visible_weights], axis=2)
        if results.get('keypoints_3d') is not None:
            results['keypoints_3d'] = keypoints
            results['lifting_target'] = keypoints[results['target_idx']]
            results['lifting_target_visible'] = keypoints_visible[
                results['target_idx']]
        if self.dst_flip_indices is not None:
            results['flip_indices'] = self.dst_flip_indices
        else:
            results['flip_indices'] = flip_indices

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        if self.src is not None and self.dst is not None:
            repr_str += f'(src={self.src!r}, dst={self.dst!r}, '\
                        f'num_keypoints={self.num_keypoints})'
        else:
            repr_str += f'(num_keypoints={self.num_keypoints}, '\
                        f'mapping={self.mapping})'
        return repr_str


@TRANSFORMS.register_module()
class SingleHandConverter(BaseTransform):
    """Mapping a single hand keypoints into double hands according to the given
    mapping and hand type.

    Required Keys:

        - keypoints
        - keypoints_visible
        - hand_type

    Modified Keys:

        - keypoints
        - keypoints_visible

    Args:
        num_keypoints (int): The number of keypoints in target dataset.
        left_hand_mapping (list): A list containing mapping indexes. Each
            element has format (source_index, target_index)
        right_hand_mapping (list): A list containing mapping indexes. Each
            element has format (source_index, target_index)

    Example:
        >>> import numpy as np
        >>> self = SingleHandConverter(
        >>>     num_keypoints=42,
        >>>     left_hand_mapping=[
        >>>         (0, 0), (1, 1), (2, 2), (3, 3)
        >>>     ],
        >>>     right_hand_mapping=[
        >>>         (0, 21), (1, 22), (2, 23), (3, 24)
        >>>     ])
        >>> results = dict(
        >>>     keypoints=np.arange(84).reshape(2, 21, 2),
        >>>     keypoints_visible=np.arange(84).reshape(2, 21, 2) % 2,
        >>>     hand_type=np.array([[0, 1], [1, 0]]))
        >>> results = self(results)
    """

    def __init__(self, num_keypoints: int,
                 left_hand_mapping: Union[List[Tuple[int, int]],
                                          List[Tuple[Tuple, int]]],
                 right_hand_mapping: Union[List[Tuple[int, int]],
                                           List[Tuple[Tuple, int]]]):
        self.num_keypoints = num_keypoints
        self.left_hand_converter = KeypointConverter(num_keypoints,
                                                     left_hand_mapping)
        self.right_hand_converter = KeypointConverter(num_keypoints,
                                                      right_hand_mapping)

    def transform(self, results: dict) -> dict:
        """Transforms the keypoint results to match the target keypoints."""
        assert 'hand_type' in results, (
            'hand_type should be provided in results')
        hand_type = results['hand_type']

        if np.sum(hand_type - [[0, 1]]) <= 1e-6:
            # left hand
            results = self.left_hand_converter(results)
        elif np.sum(hand_type - [[1, 0]]) <= 1e-6:
            results = self.right_hand_converter(results)
        else:
            raise ValueError('hand_type should be left or right')

        return results

    def __repr__(self) -> str:
        """print the basic information of the transform.

        Returns:
            str: Formatted string.
        """
        repr_str = self.__class__.__name__
        repr_str += f'(num_keypoints={self.num_keypoints}, '\
                    f'left_hand_converter={self.left_hand_converter}, '\
                    f'right_hand_converter={self.right_hand_converter})'
        return repr_str
