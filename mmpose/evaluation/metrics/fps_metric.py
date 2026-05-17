# Copyright (c) OpenMMLab. All rights reserved.
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from mmpose.registry import METRICS


def _find_runner_in_call_stack():
    """Walk the Python call stack to find the enclosing mmengine Runner.

    When a ``BaseMetric.process()`` is called during testing the call chain
    is always::

        TestLoop.run_iter
          → Evaluator.process
            → BaseMetric.process  (← we are here)

    The ``TestLoop`` (or ``ValLoop``) instance sitting higher in the stack
    carries a ``runner`` attribute.  We locate it by looking for any frame
    whose ``self`` has both a ``runner`` and an ``evaluator`` attribute, which
    is the unique signature of a Loop object.
    """
    frame = sys._getframe(1)
    while frame is not None:
        local_self = frame.f_locals.get('self')
        if local_self is not None:
            runner = getattr(local_self, 'runner', None)
            evaluator = getattr(local_self, 'evaluator', None)
            if (runner is not None and evaluator is not None
                    and hasattr(runner, 'model')):
                return runner
        frame = frame.f_back
    return None


def _make_forward_wrapper(original_forward, metric: 'FPS'):
    """Wrap a model's forward method to record per-batch inference time.

    Uses CUDA Events for GPU timing (sub-millisecond accuracy) and falls back
    to ``time.perf_counter`` on CPU-only setups.  The first
    ``metric._warmup_batches`` calls are forwarded without recording so that
    JIT compilation and GPU warm-up do not skew measurements.
    """

    def timed_forward(*args, **kwargs):
        metric._batch_counter += 1
        if metric._batch_counter <= metric._warmup_batches:
            return original_forward(*args, **kwargs)

        if torch.cuda.is_available():
            # Drain any pending GPU work before the start marker so the
            # event truly captures only this forward pass.
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = original_forward(*args, **kwargs)
            end.record()
            # Block until the kernel completes so elapsed_time() is valid.
            torch.cuda.synchronize()
            metric._inference_times.append(
                start.elapsed_time(end) / 1000.0)  # ms → s
        else:
            t0 = time.perf_counter()
            result = original_forward(*args, **kwargs)
            metric._inference_times.append(time.perf_counter() - t0)

        return result

    return timed_forward


@METRICS.register_module()
class FPS(BaseMetric):
    """Inference FPS (frames per second) evaluation metric.

    Measures the throughput of the model's ``forward`` pass only, excluding
    data loading, image decoding, affine transforms, data pre-processing
    (normalisation / device transfer), and metric computation.

    Timing is performed by monkey-patching a method on the model with a thin
    wrapper that records CUDA Event durations (GPU) or ``time.perf_counter``
    durations (CPU).  The wrapper is installed lazily on the first call to
    :meth:`process` by locating the active mmengine ``Runner`` via the
    Python call stack.  It is automatically removed after
    :meth:`compute_metrics` so the model is left unmodified.

    The patched method is chosen as follows:

    - If the model exposes ``_inference_forward`` (the convention used by
      external wrappers such as ``Sapiens2PoseEstimator``,
      ``PETRPoseEstimator``, and ``PCTPoseEstimator``), that method is patched.
      It should contain *only* the inner neural-network call, excluding
      post-processing such as heatmap decoding or coordinate transforms.
    - Otherwise ``forward`` is patched (the behaviour for native MMPose models
      like ``TopdownPoseEstimator``).

    ``FPS = timed_frames / total_inference_time``

    where *timed_frames* is the total number of samples processed after the
    warmup period and *total_inference_time* is the sum of the measured
    ``forward`` durations for those batches.

    Args:
        warmup_batches (int): Number of leading batches whose inference time
            is discarded to avoid GPU warm-up / JIT compilation bias.
            Default: ``5``.
        collect_device (str): Device for distributed result collection.
            Default: ``'cpu'``.
        prefix (str, optional): Metric name prefix.  Default: ``None``.
    """

    default_prefix: Optional[str] = 'fps'

    def __init__(self,
                 warmup_batches: int = 5,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self._warmup_batches = warmup_batches
        self._batch_counter: int = 0
        self._inference_times: List[float] = []
        self._original_forward = None
        self._model = None
        self._patch_target: str = 'forward'

    def _install_wrapper(self, runner) -> None:
        """Monkey-patch the model's timing target with the timing wrapper.

        For native MMPose models (e.g. ``TopdownPoseEstimator``) we patch
        ``forward``, which covers the full backbone + head computation.

        External wrappers (``Sapiens2PoseEstimator``, ``PETRPoseEstimator``,
        ``PCTPoseEstimator``) expose a ``_inference_forward`` method that
        contains *only* the inner model call, excluding post-processing steps
        such as heatmap decoding and coordinate transforms.  When present,
        ``_inference_forward`` is patched instead so that the measured time
        more closely reflects pure neural-network compute.
        """
        model = runner.model
        # Unwrap DP / DDP so we patch the underlying module's method.
        if hasattr(model, 'module'):
            model = model.module
        self._model = model

        if hasattr(model, '_inference_forward'):
            self._patch_target = '_inference_forward'
            self._original_forward = model._inference_forward
            model._inference_forward = _make_forward_wrapper(
                self._original_forward, self)
        else:
            self._patch_target = 'forward'
            self._original_forward = model.forward
            model.forward = _make_forward_wrapper(
                self._original_forward, self)

    def _remove_wrapper(self) -> None:
        """Restore the original method."""
        if self._model is not None and self._original_forward is not None:
            setattr(self._model, self._patch_target, self._original_forward)
            self._original_forward = None
            self._model = None

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Count samples and lazily install the timing wrapper.

        Args:
            data_batch (Sequence[dict]): Raw batch from the dataloader
                (unused beyond triggering wrapper installation).
            data_samples (Sequence[dict]): Model predictions for this batch.
        """
        # Install the timing wrapper once, on the very first process() call.
        # We locate the Runner by inspecting the call stack: the enclosing
        # TestLoop/ValLoop frame has both .runner and .evaluator attributes.
        if self._original_forward is None:
            runner = _find_runner_in_call_stack()
            if runner is not None:
                self._install_wrapper(runner)

        self.results.append({'n_samples': len(data_samples)})

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute FPS from accumulated timings.

        Args:
            results (list): List of dicts with ``n_samples`` keys, one per
                processed batch (collected across all ranks).

        Returns:
            Dict[str, float]: ``{'FPS': <value>}``.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        self._remove_wrapper()

        total_samples = sum(r['n_samples'] for r in results)
        # Warmup samples: first _warmup_batches batches were not timed.
        # Subtract their sample count from total_samples so the denominator
        # matches the timed frames only.
        warmup_results = results[:self._warmup_batches]
        warmup_samples = sum(r['n_samples'] for r in warmup_results)
        timed_samples = total_samples - warmup_samples

        total_time = sum(self._inference_times)

        logger.info(
            f'FPS metric: {len(self._inference_times)} timed batches, '
            f'{timed_samples} timed frames, '
            f'{total_time:.3f}s total inference time.')

        if total_time > 0 and timed_samples > 0:
            fps = timed_samples / total_time
        else:
            fps = 0.0
            logger.warning(
                'FPS could not be computed: no timed batches recorded. '
                'This may happen if the dataset has fewer batches than '
                f'warmup_batches={self._warmup_batches}.')

        return {'FPS': fps}
