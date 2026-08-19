# Copyright (c) OpenMMLab. All rights reserved.
"""Post-processing pipeline: chains multiple filters over pose predictions."""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Union

from mmengine.config import Config

from mmpose.structures import PoseDataSample

from .base import BaseFilter, sequence_key_from_path
from .registry import POST_PROCESS_FILTERS


class PostProcessingPipeline:
    """Ordered chain of :class:`BaseFilter` instances applied to pose predictions.

    The pipeline splits the filter list at the first offline filter and
    operates in two modes:

    **All-online mode** - :meth:`process` applies every filter in order and
    returns the processed :class:`PoseDataSample` immediately.  Per-frame
    timing is accumulated in ``_frame_times``.  Sequence boundaries are
    detected from ``img_path``; filter state is reset at each new sequence.

    **Any-offline mode** - :meth:`process` runs the online prefix (if any),
    strips ``img`` when the pipeline needs images, buffers the pose-only
    result, and returns ``None``.  :meth:`evaluate` runs the offline suffix
    over the collected buffer and returns the list of processed frames.
    Online-prefix time is accumulated per frame; offline-suffix time is
    recorded as a single :meth:`evaluate` call.

    **Identity (empty filters)** - a pipeline with no filters is online and
    :meth:`process` returns each frame unchanged.  This is the intended way
    to re-run evaluation on a saved prediction bundle (different metrics or
    settings) without transforming the predictions.

    Args:
        filters: Ordered list of :class:`BaseFilter` instances.  An empty
            list is a valid identity pipeline.
        needs_images: Declare that this pipeline consumes frame images.
            Must be set to ``True`` in the pipeline config whenever any
            filter has ``requires_images=True`` (validated here, at build
            time).  When set, the driver attaches each frame's pixels as a
            data field ``ds.img`` before :meth:`process` (see
            :class:`BaseFilter` for the exact contract) and this pipeline
            strips ``img`` before buffering or returning so downstream
            consumers don't pin the pixel arrays in memory.  Image-requiring
            filters must be online and must appear before the first offline
            filter; offline filters never see frame images.
    """

    def __init__(
        self,
        filters: List[BaseFilter],
        needs_images: bool = False,
    ) -> None:
        self.filters = filters
        # Vacuous all([]) is True, but spell out the identity case so an
        # empty filter list is never treated as offline by accident.
        self.is_online: bool = (not filters) or all(f.online for f in filters)
        self.needs_images: bool = bool(needs_images)

        # Split at the first offline filter: online prefix runs in
        # process(); offline suffix (and any later online filters) run in
        # evaluate() via process_sequence.
        split = len(filters)
        for i, f in enumerate(filters):
            if not f.online:
                split = i
                break
        self._online_filters: List[BaseFilter] = filters[:split]
        self._offline_filters: List[BaseFilter] = filters[split:]

        image_filters = [
            f for f in filters if getattr(f, 'requires_images', False)
        ]
        if image_filters and not self.needs_images:
            names = [type(f).__name__ for f in image_filters]
            raise ValueError(
                f'Filters {names} require frame images, but the '
                f'pipeline config does not declare needs_images=True. Add '
                f"needs_images=True to the post_processor dict of the "
                f'post-processing config.')

        bad_offline_image = [
            type(f).__name__ for f in image_filters if not f.online
        ]
        if bad_offline_image:
            raise ValueError(
                f'Filters {bad_offline_image} require frame images but are '
                f'offline. Image-requiring filters must be online so they '
                f'can consume ds.img during process() before results are '
                f'buffered without pixels.')

        bad_after_offline = [
            type(f).__name__ for f in self._offline_filters
            if getattr(f, 'requires_images', False)
        ]
        if bad_after_offline:
            raise ValueError(
                f'Filters {bad_after_offline} require frame images but '
                f'appear at or after the first offline filter. Move '
                f'image-requiring filters into the online prefix (before '
                f'any offline filter); offline stages never see ds.img.')

        # Online runtime bookkeeping
        self._frame_times: List[float] = []
        self._current_seq: Optional[str] = None

        # Offline buffer
        self._buffer: List[PoseDataSample] = []
        # Retained after evaluate() clears the buffer so timing accessors
        # still know how many frames were processed.
        self._num_buffered: int = 0

        # Timing: online-prefix total is snapshotted before evaluate()
        # clears _frame_times; offline-suffix total is set by evaluate().
        self._online_total_s: float = 0.0
        self._eval_total_s: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        ds: PoseDataSample,
    ) -> Optional[PoseDataSample]:
        """Process one frame.

        All-online pipeline: applies all filters and returns the result.
        Any-offline pipeline: runs the online prefix (if any), strips
        ``img`` when needed, buffers the result, and returns ``None``.

        Args:
            ds: Prediction-only :class:`PoseDataSample` for this frame.

        Returns:
            Processed :class:`PoseDataSample` (all-online) or ``None``
            (any-offline).
        """
        if not self.is_online:
            return self._process_hybrid(ds)

        img_path = ds.metainfo.get('img_path', '')
        seq_key = sequence_key_from_path(img_path)

        if self._current_seq is None:
            self._current_seq = seq_key
        elif seq_key != self._current_seq:
            for f in self.filters:
                f.reset()
            self._current_seq = seq_key

        t0 = time.perf_counter()
        result = ds
        for f in self.filters:
            result = f.process_frame(result, seq_key)
        self._frame_times.append(time.perf_counter() - t0)

        if self.needs_images and result is not None:
            # ds.new() copies data fields by reference, so the returned
            # sample would otherwise keep the frame's pixels alive in the
            # caller's results list for the whole run.
            result.pop('img', None)

        return result

    def evaluate(self) -> List[PoseDataSample]:
        """Finalise and return all processed frames.

        All-online pipeline: returns an empty list (frames were already
        returned by successive :meth:`process` calls).  This method is a
        no-op in the timing sense for the online path; it just resets
        state.

        Any-offline pipeline: runs the offline suffix over the buffer
        (online prefix already ran in :meth:`process`), records timing,
        and clears the buffer.

        Returns:
            List of processed :class:`PoseDataSample`s in input order.
        """
        if self.is_online:
            # State reset for re-use; frames were already returned by process()
            self._reset_online_state()
            return []

        # Snapshot online-prefix time before _reset_online_state clears it.
        self._online_total_s = sum(self._frame_times)
        self._num_buffered = len(self._buffer)

        t0 = time.perf_counter()
        frames = list(self._buffer)
        for f in self._offline_filters:
            frames = f.process_sequence(frames)
        self._eval_total_s = time.perf_counter() - t0

        self._buffer.clear()
        self._reset_online_state()
        return frames

    def reset(self) -> None:
        """Reset all internal state.  Useful between dataset runs."""
        for f in self.filters:
            f.reset()
        self._reset_online_state()
        self._buffer.clear()
        self._num_buffered = 0
        self._online_total_s = 0.0
        self._eval_total_s = 0.0

    # ------------------------------------------------------------------
    # Timing accessors
    # ------------------------------------------------------------------

    @property
    def per_frame_ms(self) -> float:
        """Average per-frame processing time in milliseconds (online prefix)."""
        if not self._frame_times:
            return 0.0
        return 1000.0 * sum(self._frame_times) / len(self._frame_times)

    @property
    def total_s(self) -> float:
        """Total processing time in seconds.

        All-online: sum of per-frame times accumulated so far.
        Any-offline: online-prefix time plus the last :meth:`evaluate`
        call.  Before :meth:`evaluate`, the online prefix is still in
        ``_frame_times``; afterwards it is snapshotted in
        ``_online_total_s``.
        """
        if self.is_online:
            return sum(self._frame_times)
        online_s = (
            sum(self._frame_times) if self._frame_times
            else self._online_total_s)
        return online_s + self._eval_total_s

    @property
    def num_frames(self) -> int:
        """Number of frames processed (online) or buffered (offline)."""
        if self.is_online:
            return len(self._frame_times)
        if self._buffer:
            return len(self._buffer)
        return self._num_buffered

    def perf_dict(self) -> Dict[str, float]:
        """Return a flat dict of performance metrics for serialisation."""
        n = self.num_frames
        total = self.total_s
        return {
            'postproc/latency_ms_per_frame': (
                1000.0 * total / n if n > 0 else 0.0),
            'postproc/total_s': total,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _process_hybrid(self, ds: PoseDataSample) -> None:
        """Run online prefix, strip images, and buffer for evaluate()."""
        img_path = ds.metainfo.get('img_path', '')
        seq_key = sequence_key_from_path(img_path)

        if self._current_seq is None:
            self._current_seq = seq_key
        elif seq_key != self._current_seq:
            for f in self._online_filters:
                f.reset()
            self._current_seq = seq_key

        t0 = time.perf_counter()
        result = ds
        for f in self._online_filters:
            result = f.process_frame(result, seq_key)
        self._frame_times.append(time.perf_counter() - t0)

        if self.needs_images and result is not None:
            result.pop('img', None)

        self._buffer.append(result)
        return None

    def _reset_online_state(self) -> None:
        self._frame_times = []
        self._current_seq = None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def build_post_processor(
    cfg: Union[Dict, str, Config],
) -> PostProcessingPipeline:
    """Build a :class:`PostProcessingPipeline` from a config.

    ``cfg`` may be:

    * A ``dict`` with key ``'post_processor'`` mapping to a pipeline config,
      or directly a pipeline config dict with ``type='PostProcessingPipeline'``.
    * A file path string pointing to a Python config file that defines a
      ``post_processor`` variable.
    * An :class:`~mmengine.config.Config` instance with a ``post_processor``
      attribute.

    Each filter in ``filters`` is built from :data:`POST_PROCESS_FILTERS`.
    ``filters`` may be omitted or set to ``[]`` for an identity pipeline.

    Example config file::

        post_processor = dict(
            type='PostProcessingPipeline',
            filters=[
                dict(type='OKSTracker', match_thr=0.5),
                dict(type='OneEuroSmoother', min_cutoff=0.004, beta=0.7),
            ],
        )

    Pipelines whose filters read frame images must declare it explicitly.
    Image-requiring filters may be followed by offline pose-only filters
    (e.g. a CNN tracker then SmoothNet)::

        post_processor = dict(
            type='PostProcessingPipeline',
            needs_images=True,
            filters=[...],
        )
    """
    if isinstance(cfg, str):
        cfg = Config.fromfile(cfg)

    if isinstance(cfg, Config):
        pipeline_cfg = cfg.get('post_processor')
        if pipeline_cfg is None:
            raise KeyError(
                "Config file must define a top-level 'post_processor' variable.")
    elif isinstance(cfg, dict):
        if 'post_processor' in cfg:
            pipeline_cfg = cfg['post_processor']
        else:
            pipeline_cfg = cfg
    else:
        raise TypeError(f'Unsupported config type: {type(cfg)}')

    pipeline_cfg = dict(pipeline_cfg)

    # Build filters
    filter_cfgs = pipeline_cfg.pop('filters', [])
    filters: List[BaseFilter] = []
    for fc in filter_cfgs:
        filters.append(POST_PROCESS_FILTERS.build(fc))

    # Ignore 'type' key if present
    pipeline_cfg.pop('type', None)

    return PostProcessingPipeline(filters=filters, **pipeline_cfg)
