"""Wrapper integrating OpenPifPaf (pose + spatio-temporal association) from
``external/openpifpaf`` into MMPose v1.x for evaluation.

OpenPifPaf (`Kreiss et al., T-ITS 2021
<https://ieeexplore.ieee.org/document/9617128>`_) is a **bottom-up** method.
Its keypoints are grouped within a frame by Composite Association Fields
(CAF); tracking adds a *temporal* composite field (Tcaf) that associates
joints between neighbouring frames, so track IDs come out of the same
greedy decoder that produces the poses -- there is no separate matcher.

Two pieces of upstream state make this strictly sequential, and both are
used here unmodified:

* ``network.TrackingBase`` wraps a single-image backbone with a
  ``RunningCache`` over frames ``[0, -1]``: each call contributes the
  current frame and the heads see it concatenated with the previous one.
* ``decoder.TrackingPose`` (subclass of ``decoder.TrackBase``) owns the
  pool of active tracks and the frame counter.

Both subscribe to openpifpaf's ``Signal('eval_reset')``, which is exactly
what :meth:`reset_tracking` emits at each sequence boundary.

Nothing about the network or the decoder is reimplemented here.  The
preprocessing chain is upstream's own ``transforms`` objects, assembled to
match ``Posetrack2018.common_eval_preprocess`` at ``batch_size == 1``
(``RescaleAbsolute(long_edge=801)`` then ``CenterPadTight(16)`` then
``EVAL_TRANSFORM``), and coordinates are mapped back with upstream's own
``Annotation.inverse_transform(meta)``.  This wrapper only translates
interfaces:

- **Image hand-off.**  MMPose's bottom-up path hands the wrapper a batched
  tensor, so the config's pipeline deliberately does *no* resizing and its
  ``data_preprocessor`` does *no* normalisation: the tensor still holds the
  raw uint8 BGR pixels, which are converted back to a PIL RGB image and
  fed to upstream's transform chain.  Doing the geometry here instead
  would mean reimplementing ``RescaleAbsolute``/``CenterPadTight`` and
  their ``meta`` bookkeeping, and any drift would be silent.
- **Checkpoint loading.**  The released ``.pkl`` files are pickled *whole
  model objects*, not state dicts, so ``network.Factory`` loads them.  The
  wrapper is registered in ``mmpose.apis.inference.CUSTOM_POSE_WRAPPER_TYPES``
  so ``init_model(config, checkpoint)`` forwards the path instead of
  calling MMEngine's ``load_checkpoint``.
- **Keypoint layout.**  The tracking checkpoints are trained on
  PoseTrack-2018's 17-joint layout (``nose, head_bottom, head_top,
  left/right_ear, ...``).  ``map_to_coco=True`` reprojects onto COCO-17 via
  ``KeypointConverter(src='posetrack18', dst='coco')``; ``left_eye`` /
  ``right_eye`` have no PoseTrack counterpart and stay at zero confidence.
  The COCO checkpoints (``shufflenetv2k16``/``k30``) are already COCO-17,
  so they set ``map_to_coco=False``.

Checkpoint availability, verified 2026-08-08: **``tshufflenetv2k30`` is the
only released tracking checkpoint.**  Upstream marks ``tshufflenetv2k16``
as ``PRETRAINED_UNAVAILABLE`` (``plugins/posetrack/__init__.py:9``) even
though its own benchmark script lists it as the default, so no config for
it exists here.

Usage example (in a config file)::

    emits_track_ids = True   # routes tools/benchmark_e2e.py to run_tracking

    model = dict(
        type='OpenPifPafPoseEstimator',
        openpifpaf_root='external/openpifpaf',
        decoder='trackingpose:0',
        long_edge=801,
        map_to_coco=True,
        data_preprocessor=dict(type='PoseDataPreprocessor', bgr_to_rgb=False),
    )
"""

import os
import sys
from typing import List, Optional

import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmpose.registry import MODELS
from mmpose.utils.typing import SampleList

_OPENPIFPAF_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'external',
                 'openpifpaf'))


def _ensure_openpifpaf_importable(root: str) -> str:
    """Make ``openpifpaf`` importable, preferring an already-installed copy.

    An openpifpaf that is already importable is left alone: its compiled
    decoder was built for the interpreter that installed it, whereas the
    submodule's was built for whichever interpreter last ran
    ``pip install -e external/openpifpaf`` there.  Those can differ, because
    upstream's setup.py builds the extension with
    ``no_python_abi_suffix=True`` (setup.py:56) -- the artifact is plain
    ``src/openpifpaf/_cpp.so`` with no ``cpython-3X`` tag, so one checkout
    holds exactly one build and a second env cannot add its own without
    overwriting the first.  Prepending the submodule path unconditionally
    would therefore shadow a correctly-built install with an ABI-mismatched
    ``.so``.

    Only when openpifpaf is not installed at all does the submodule tree go
    on ``sys.path``, which is the editable-install case.

    Returns the resolved root, or ``''`` when an installed copy is used.
    """
    try:
        import openpifpaf  # noqa: F401
        return ''
    except ImportError:
        pass

    root = os.path.abspath(root)
    src = os.path.join(root, 'src')
    if not os.path.isdir(src):
        raise FileNotFoundError(
            f'openpifpaf is not installed and its sources are not at '
            f'{src!r}. It is vendored as a git submodule; run '
            f'`git submodule update --init external/openpifpaf`, then build '
            f'its C++ decoder extension with '
            f'`pip install -e external/openpifpaf --no-build-isolation '
            f'--no-deps` (the CifCaf/Tcaf decoder *is* that extension, so '
            f'the build is not optional). To share one checkout across two '
            f'environments, install it non-editably in the second '
            f'(`pip install ./external/openpifpaf ...`) so each gets its own '
            f'_cpp.so.')
    if src not in sys.path:
        sys.path.insert(0, src)
    return root


@MODELS.register_module()
class OpenPifPafPoseEstimator(BaseModel):
    """MMPose v1.x wrapper for OpenPifPaf, with or without tracking.

    Args:
        checkpoint (str): OpenPifPaf checkpoint -- a local ``.pkl`` path or
            a shortcut name from ``openpifpaf.CHECKPOINT_URLS`` (e.g.
            ``'tshufflenetv2k30'``, downloaded to the torch hub cache).
            ``tools/benchmark_e2e.py``'s ``pose_checkpoint`` argument is
            forwarded into this field; see
            ``mmpose.apis.inference.CUSTOM_POSE_WRAPPER_TYPES``.
        openpifpaf_root (str): Path to the ``external/openpifpaf`` checkout.
            Only used when ``openpifpaf`` is not already installed; see
            :func:`_ensure_openpifpaf_importable`.
        decoder (str, optional): Decoder request in upstream's CLI syntax,
            e.g. ``'trackingpose:0'`` (the paper's method),
            ``'posesimilarity:0'`` (the paper's own tracking baseline) or
            ``'cifcaf:0'`` (single-image, no tracking).  ``None`` lets
            upstream pick by priority.
        long_edge (int, optional): Long-side rescale before padding.  ``801``
            reproduces ``Posetrack2018.eval_long_edge``, the authors' eval
            setting.  ``None`` disables rescaling and runs at native
            resolution.
        map_to_coco (bool): Reproject a PoseTrack-17 output onto COCO-17.
            Set ``False`` for the COCO-trained checkpoints.
        device (str): Set by ``init_model``; the upstream decoder is CPU-side
            so only the network is moved.
        data_preprocessor (dict, optional): Must **not** normalise -- the
            wrapper hands raw pixels to upstream's own transform chain.
        init_cfg (dict, optional): MMEngine init config.
    """

    def __init__(
        self,
        checkpoint: str,
        openpifpaf_root: str = _OPENPIFPAF_ROOT,
        decoder: Optional[str] = None,
        long_edge: Optional[int] = 801,
        map_to_coco: bool = True,
        device: str = 'cpu',
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ):
        if data_preprocessor is not None and isinstance(data_preprocessor,
                                                        dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        _ensure_openpifpaf_importable(openpifpaf_root)

        import openpifpaf
        import openpifpaf.plugin
        from openpifpaf import decoder as opp_decoder
        from openpifpaf import network as opp_network
        from openpifpaf import transforms as opp_transforms

        # Registers the posetrack plugin, which is what defines the
        # Tcaf head meta and the tshufflenetv2k* checkpoint URLs.
        openpifpaf.plugin.register()

        resolved = openpifpaf.CHECKPOINT_URLS.get(checkpoint, checkpoint)
        if resolved is openpifpaf.PRETRAINED_UNAVAILABLE:
            raise ValueError(
                f'OpenPifPaf checkpoint {checkpoint!r} is marked '
                f'PRETRAINED_UNAVAILABLE upstream: the authors never '
                f'released those weights. Use "tshufflenetv2k30", the only '
                f'released tracking checkpoint.')

        # `decoder.Factory` is not re-exported from openpifpaf.decoder, so
        # reach it through the submodule the same way decoder.cli does.
        from openpifpaf.decoder.factory import Factory as DecoderFactory

        if decoder is not None:
            DecoderFactory.decoder_request_from_args([decoder])
        self.decoder_request = decoder

        opp_network.Factory.checkpoint = checkpoint
        self.opp_model, _ = opp_network.Factory().factory()
        self.opp_model.eval()
        self.processor = opp_decoder.factory(self.opp_model.head_metas)

        # Upstream's Posetrack2018.common_eval_preprocess at batch_size==1,
        # minus the annotation-side steps (there is no GT to transform here).
        steps = [opp_transforms.NormalizeAnnotations()]
        if long_edge:
            steps.append(opp_transforms.RescaleAbsolute(long_edge))
        steps.append(opp_transforms.CenterPadTight(16))
        steps.append(opp_transforms.EVAL_TRANSFORM)
        self.preprocess = opp_transforms.Compose(steps)

        self.long_edge = long_edge
        self._device = device
        self.map_to_coco = map_to_coco
        self.keypoint_converter = None
        if map_to_coco:
            from mmpose.datasets.transforms import KeypointConverter
            self.keypoint_converter = KeypointConverter(
                src='posetrack18', dst='coco')

        self.num_keypoints = 17

    # ── Tracking state ─────────────────────────────────────────────────

    def reset_tracking(self) -> None:
        """Clear the backbone feature cache and the decoder's track pool.

        Emits upstream's ``eval_reset`` signal, which is the same mechanism
        ``openpifpaf.eval`` uses between videos: ``TrackingBase.reset``
        drops the ``RunningCache``, and ``TrackBase.reset`` clears
        ``self.active`` and zeroes ``frame_number``.  Harmless for the
        non-tracking decoders, which do not subscribe.
        """
        from openpifpaf.signal import Signal

        Signal.emit('eval_reset')

    # ── Inference ──────────────────────────────────────────────────────

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        if mode == 'tensor':
            return self._inference_forward(inputs)
        raise NotImplementedError(
            f'{type(self).__name__} is inference-only; mode={mode!r} is not '
            f'supported.')

    def _inference_forward(self, inputs: torch.Tensor,
                           img_shapes: Optional[List[tuple]] = None):
        """Preprocess + network + decoder for a batch of raw-pixel tensors.

        Kept separate from :meth:`predict` so the ``FPS`` metric can time
        the model without the InstanceData packing, as the other wrappers
        in this package do.  Returns ``(annotations, metas)`` per image.

        Images are decoded one at a time regardless of batch size.  For the
        tracking decoders that is mandatory (the ``RunningCache`` pairs each
        frame with its predecessor); for the single-image decoders it is
        merely how upstream's ``Processor`` is driven at batch size 1, which
        is also what ``CenterPadTight`` assumes.
        """
        preds: List[list] = []
        metas: List[dict] = []
        for i, raw in enumerate(inputs):
            shape = img_shapes[i] if img_shapes is not None else None
            image, meta = self._preprocess_one(raw, shape)
            annotations = self.processor.batch(
                self.opp_model, image.unsqueeze(0), device=self._device)[0]
            preds.append(annotations)
            metas.append(meta)
        return preds, metas

    def _preprocess_one(self, raw: torch.Tensor,
                        img_shape: Optional[tuple] = None):
        """Raw ``(3, H, W)`` BGR pixel tensor -> upstream-preprocessed image.

        The config's data_preprocessor is configured not to normalise, so
        ``raw`` still holds integral 0..255 values (exactly representable in
        float32) and this round trip is lossless.

        ``img_shape`` is the sample's true ``(H, W)``.  When a batch mixes
        resolutions -- EMDB has both 1440x1920 and 720x960 sequences --
        MMEngine's ``stack_batch`` zero-pads every image up to the batch
        maximum, and that padding must be removed before
        ``RescaleAbsolute`` sizes the long edge, or the smaller images would
        be rescaled by the wrong factor and every coordinate would come back
        shifted.
        """
        import PIL.Image

        if img_shape is not None:
            h, w = int(img_shape[0]), int(img_shape[1])
            raw = raw[:, :h, :w]

        bgr = raw.detach().to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        pil = PIL.Image.fromarray(bgr[:, :, ::-1])

        meta = {
            'offset': np.array([0.0, 0.0]),
            'scale': np.array([1.0, 1.0]),
            'valid_area': np.array([0.0, 0.0, pil.size[0], pil.size[1]]),
        }
        image, _, meta = self.preprocess(pil, [], meta)
        return image, meta

    def predict(self, inputs: torch.Tensor,
                data_samples: SampleList) -> SampleList:
        """Run one frame (or a batch of them) and pack MMPose predictions.

        ``run_tracking`` in ``tools/benchmark_e2e.py`` calls this one frame
        at a time and in dataset order, which is what the ``RunningCache``
        and the decoder's track pool require.
        """
        img_shapes = [
            ds.metainfo.get('img_shape', ds.metainfo.get('ori_shape'))
            for ds in data_samples
        ]
        preds, metas = self._inference_forward(inputs, img_shapes)

        for annotations, meta, ds in zip(preds, metas, data_samples):
            annotations = [ann.inverse_transform(meta) for ann in annotations]
            ds.pred_instances = self._pack(annotations)
        return data_samples

    def _pack(self, annotations: list) -> InstanceData:
        """Upstream ``Annotation`` list -> MMPose ``InstanceData``."""
        n = len(annotations)
        n_src = (len(annotations[0].data) if n else self.num_keypoints)

        keypoints = np.zeros((n, n_src, 2), dtype=np.float32)
        scores = np.zeros((n, n_src), dtype=np.float32)
        bboxes = np.zeros((n, 4), dtype=np.float32)
        bbox_scores = np.zeros(n, dtype=np.float32)
        track_ids = np.full(n, -1, dtype=np.int32)

        for i, ann in enumerate(annotations):
            data = np.asarray(ann.data, dtype=np.float32)  # (K, 3): x, y, v
            keypoints[i] = data[:, :2]
            scores[i] = data[:, 2]
            x, y, w, h = ann.bbox()
            bboxes[i] = [x, y, x + w, y + h]
            bbox_scores[i] = float(ann.score)
            # Set by TrackBase.annotations(); absent for non-tracking
            # decoders, which leaves the id at -1.
            if getattr(ann, 'id_', None) is not None:
                track_ids[i] = int(ann.id_)

        if self.keypoint_converter is not None:
            keypoints, scores = self._to_coco(keypoints, scores)

        pred = InstanceData()
        pred.keypoints = keypoints
        pred.keypoint_scores = scores
        pred.keypoints_visible = (scores > 0).astype(np.float32)
        pred.bboxes = bboxes
        pred.bbox_scores = bbox_scores
        pred.track_ids = track_ids
        return pred

    def _to_coco(self, keypoints: np.ndarray, scores: np.ndarray):
        """PoseTrack-17 -> COCO-17 via the shared KeypointConverter table.

        Only the 15 COCO joints with a same-named PoseTrack counterpart are
        filled; ``left_eye``/``right_eye`` stay at zero confidence rather
        than being synthesised, since the tracking decoder already treats
        the ear/eye region as unreliable (it excludes the ears from pose
        similarity, ``decoder/tracking_pose.py:41-46``).
        """
        n = keypoints.shape[0]
        out_kpts = np.zeros((n, self.num_keypoints, 2), dtype=np.float32)
        out_scores = np.zeros((n, self.num_keypoints), dtype=np.float32)
        if n == 0:
            return out_kpts, out_scores

        mapping = self.keypoint_converter.mapping
        src_idx = [m[0] for m in mapping]
        dst_idx = [m[1] for m in mapping]
        out_kpts[:, dst_idx] = keypoints[:, src_idx]
        out_scores[:, dst_idx] = scores[:, src_idx]
        return out_kpts, out_scores
