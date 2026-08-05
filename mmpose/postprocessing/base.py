# Copyright (c) OpenMMLab. All rights reserved.
"""Base class for pose post-processing filters."""

from __future__ import annotations

import os.path as osp
from abc import ABCMeta, abstractmethod
from typing import List, Optional

from mmpose.structures import PoseDataSample


def sequence_key_from_path(img_path: str) -> str:
    """Derive a sequence identifier from an image path.

    For EMDB paths like ``P1/14_outdoor_climb/images/00042.jpg`` the key is
    ``P1/14_outdoor_climb``.  For any path with fewer than three components
    the parent directory is used.  This is used to detect sequence boundaries
    so per-track filter state can be reset.
    """
    if not img_path:
        return ''
    parts = img_path.replace('\\', '/').split('/')
    # Strip the filename, then the images/ subdirectory if present
    dirs = parts[:-1]
    if dirs and dirs[-1] == 'images':
        dirs = dirs[:-1]
    return '/'.join(dirs) if dirs else ''


class BaseFilter(metaclass=ABCMeta):
    """Base class for a single-stage post-processing filter.

    Subclasses set ``online = True`` (causal, one frame at a time) or
    ``online = False`` (offline, needs the full sequence).

    Online filters implement :meth:`process_frame`.  The default
    :meth:`process_sequence` iterates the buffer, detects sequence changes via
    :func:`sequence_key_from_path`, calls :meth:`reset` at each boundary, and
    delegates to :meth:`process_frame`.

    Offline filters override :meth:`process_sequence` directly.

    **Image access.**  Filters that need the frame's pixels (e.g. appearance
    models) set ``requires_images = True`` -- as a class attribute, or as an
    instance attribute in ``__init__`` when the need depends on the
    configuration.  Such filters must be online and must appear before any
    offline filter in a pipeline whose config declares ``needs_images=True``
    (validated at build time).  The driver then attaches the frame as a data
    field ``ds.img`` before each :meth:`process_frame` call:

    * ``ds.img`` is a BGR ``(H, W, 3)`` ``uint8`` array in the **bundle's
      coordinate space**, i.e. its shape matches ``metainfo['ori_shape']``
      and the prediction coordinates.
    * ``ds.img`` may be absent or ``None`` when the source image could not
      be read; filters must degrade gracefully.
    * Filters must **not** retain a reference to ``ds.img`` beyond the
      current call -- the driver streams images in bounded chunks and
      releases them afterwards.  Cache embeddings/features, never pixels.
    * Offline filters never see ``ds.img``; the pipeline strips it before
      buffering for the offline suffix.
    """

    online: bool = True

    # Whether this filter reads ``ds.img``.  May be overridden per-instance
    # in ``__init__`` when it depends on the configuration (e.g. a tracker
    # whose appearance cost is optional).
    requires_images: bool = False

    def reset(self) -> None:
        """Reset any internal state (e.g. per-track history).

        Called automatically at sequence boundaries by
        :meth:`process_sequence` and by :class:`PostProcessingPipeline` when
        starting a new sequence in online mode.
        """

    @abstractmethod
    def process_frame(
        self,
        ds: PoseDataSample,
        seq_key: str,
    ) -> PoseDataSample:
        """Apply the filter to a single frame.

        Args:
            ds: Prediction-only :class:`PoseDataSample` for this frame.
            seq_key: Sequence identifier derived from ``img_path``; can be
                used internally if the filter manages its own state.

        Returns:
            Modified (or replaced) :class:`PoseDataSample`.
        """

    def process_sequence(
        self,
        frames: List[PoseDataSample],
    ) -> List[PoseDataSample]:
        """Apply the filter to a list of frames.

        The default implementation is suitable for online filters: iterates
        the frames in order, resets state at sequence boundaries, and calls
        :meth:`process_frame`.

        Offline filters should override this method entirely.

        Args:
            frames: Ordered list of prediction-only
                :class:`PoseDataSample`s.

        Returns:
            List of processed :class:`PoseDataSample`s (same length).
        """
        self.reset()
        current_seq: Optional[str] = None
        results: List[PoseDataSample] = []

        for ds in frames:
            img_path = ds.metainfo.get('img_path', '')
            seq_key = sequence_key_from_path(img_path)

            if current_seq is None:
                current_seq = seq_key
            elif seq_key != current_seq:
                self.reset()
                current_seq = seq_key

            results.append(self.process_frame(ds, seq_key))

        return results
