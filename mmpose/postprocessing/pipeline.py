# Copyright (c) OpenMMLab. All rights reserved.
"""Post-processing pipeline: chains multiple filters over pose predictions."""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Union

from mmengine.config import Config

from mmpose.structures import PoseDataSample

from .base import BaseFilter, sequence_key_from_path
from .registry import POST_PROCESS_FILTERS


class PostProcessingPipeline:
    """Ordered chain of :class:`BaseFilter` instances applied to pose predictions.

    The pipeline operates in two modes depending on whether all filters are
    online (causal):

    **All-online mode** - :meth:`process` applies every filter in order and
    returns the processed :class:`PoseDataSample` immediately.  Per-frame
    timing is accumulated in ``_frame_times``.  Sequence boundaries are
    detected from ``img_path``; filter state is reset at each new sequence.

    **Any-offline mode** - :meth:`process` buffers each frame and returns
    ``None``.  :meth:`evaluate` runs all filters over the collected buffer
    (chaining outputs between stages) and returns the list of processed
    frames.  :meth:`evaluate` is timed as a single call.

    Args:
        filters: Ordered list of :class:`BaseFilter` instances.
    """

    def __init__(self, filters: List[BaseFilter]) -> None:
        self.filters = filters
        self.is_online: bool = all(f.online for f in filters)

        # Online runtime bookkeeping
        self._frame_times: List[float] = []
        self._current_seq: Optional[str] = None

        # Offline buffer
        self._buffer: List[PoseDataSample] = []

        # Offline timing (set by evaluate())
        self._eval_total_s: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        ds: PoseDataSample,
    ) -> Optional[PoseDataSample]:
        """Process one frame.

        Online pipeline: applies all filters and returns the result.
        Offline pipeline: buffers the frame and returns ``None``.

        Args:
            ds: Prediction-only :class:`PoseDataSample` for this frame.

        Returns:
            Processed :class:`PoseDataSample` (online) or ``None`` (offline).
        """
        if not self.is_online:
            self._buffer.append(ds)
            return None

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

        return result

    def evaluate(self) -> List[PoseDataSample]:
        """Finalise and return all processed frames.

        Online pipeline: returns the list that was returned by successive
        :meth:`process` calls (must have been stored by the caller).  This
        method is a no-op in the timing sense for the online path; it just
        resets state.

        Offline pipeline: runs the full filter chain over the buffer, records
        timing, and clears the buffer.

        Returns:
            List of processed :class:`PoseDataSample`s in input order.
        """
        if self.is_online:
            # State reset for re-use; frames were already returned by process()
            self._reset_online_state()
            return []

        t0 = time.perf_counter()
        frames = list(self._buffer)
        for f in self.filters:
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
        self._eval_total_s = 0.0

    # ------------------------------------------------------------------
    # Timing accessors
    # ------------------------------------------------------------------

    @property
    def per_frame_ms(self) -> float:
        """Average per-frame processing time in milliseconds (online only)."""
        if not self._frame_times:
            return 0.0
        return 1000.0 * sum(self._frame_times) / len(self._frame_times)

    @property
    def total_s(self) -> float:
        """Total processing time in seconds.

        Online: sum of per-frame times accumulated so far.
        Offline: time of the last :meth:`evaluate` call.
        """
        if self.is_online:
            return sum(self._frame_times)
        return self._eval_total_s

    @property
    def num_frames(self) -> int:
        """Number of frames processed (online) or buffered (offline)."""
        if self.is_online:
            return len(self._frame_times)
        return len(self._buffer)

    def perf_dict(self) -> Dict[str, float]:
        """Return a flat dict of performance metrics for serialisation."""
        n = len(self._frame_times) if self.is_online else (
            len(self._buffer) if not self._eval_total_s
            else self.num_frames  # already cleared
        )
        return {
            'postproc/latency_ms_per_frame': self.per_frame_ms,
            'postproc/total_s': self.total_s,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

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

    Example config file::

        post_processor = dict(
            type='PostProcessingPipeline',
            filters=[
                dict(type='OKSTracker', match_thr=0.5),
                dict(type='OneEuroSmoother', min_cutoff=0.004, beta=0.7),
            ],
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
