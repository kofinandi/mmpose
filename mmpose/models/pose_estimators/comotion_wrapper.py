"""Inference wrapper for CoMotion (``external/ml-comotion``).

CoMotion (Newell et al., ICLR 2025) detects and tracks people as SMPL
bodies.  The network, the SMPL decoder, the online track handler, shot
detection, and the demo's whole-clip ``cleanup_tracks`` all run as the
unmodified upstream modules.  This file only translates images and
predictions for ``tools/benchmark_e2e.py``.

What the benchmark receives is 2D.  Upstream ``joints_coco`` (SMPL
vertices regressed to COCO-17) is projected with the demo's default
intrinsics, then written in PoseTrack-17 order.  Boxes are the demo's
SMPL-mesh boxes (5% pad).  Track ids come from ``cleanup_tracks``.

**Substitutions and gaps** (also in the config header):

- ``head_bottom`` / ``head_top`` are not CoMotion outputs.  With
  ``fill_head=True`` they are filled from the nose and shoulders using
  coefficients least-squares fitted on PoseTrack21 *train*
  (87,069 persons).  That fill is a substitute.  Set ``fill_head=False``
  to leave both joints at score 0.
- The tracker exports no confidence.  The demo's MOT file hardcodes
  confidence 1, and so does this wrapper: every emitted joint and box
  has score 1.  Keypoint AP is a single operating point.
- ``predict()`` stores the live outputs and returns an empty pose.
  ``finalize_sequence()`` runs ``cleanup_tracks`` (``min_matched_frames=1``,
  the demo value; the function default of 4 is not used) and rewrites
  those samples.  ``tools/benchmark_e2e.py`` calls it at each sequence
  boundary and after the last sequence.  Without that call the metrics
  see no people.
- Pixels are the benchmark's cv2 decode (BGR, then ``bgr_to_rgb``), not
  the demo's PIL decode.  JPEG values can differ by about a grey level.
- Transformers 5 ``LlamaDecoderLayer.forward`` returns a tensor.
  CoMotion was written for 4.48, which returns ``(hidden_states,)`` and
  is indexed with ``[0]``.  Layers inside this module are wrapped so
  they return that one-tuple.  The attention itself is unchanged.
- The test pipeline must not resize.  CoMotion crops to 512 internally
  and projects keypoints back to the original frame.

Weights are not an MMEngine state dict.  Upstream loads
``comotion_detection_checkpoint.pt`` and ``comotion_refine_checkpoint.pt``
from ``src/comotion_demo/data/`` when the module is constructed.  The
checkpoint path passed to ``init_model`` must be that detection file.
The neutral SMPL v1.1.0 body (``SMPL_NEUTRAL.pkl``) has to sit next to
``extra_smpl_reference.pt``; it is not redistributed with the code.
"""

from __future__ import annotations

import os
import sys
import time
import warnings
from typing import List, Optional, Tuple

import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmpose.registry import MODELS
from mmpose.utils.typing import SampleList

_COMOTION_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'external',
                 'ml-comotion'))

# PoseTrack21 train, 87,069 persons with nose, both shoulders, head_bottom
# and head_top labeled.  Least squares of
#   head_bottom = shoulder_mid + a * (nose - shoulder_mid)
#   head_top    = nose + b * (nose - shoulder_mid)
# Not a CoMotion output: the COCO regressor has neither joint.
_HEAD_FILL_A_BOTTOM = 0.29662723688949055
_HEAD_FILL_B_TOP = 0.3976644050600786
_HEAD_FILL_NUM_PERSONS = 87069

_NUMPY_ALIASES = {
    'bool': 'bool_',
    'int': 'int_',
    'float': 'float64',
    'complex': 'complex128',
    'object': 'object_',
    'unicode': 'str_',
    'str': 'str_',
}


def _shim_numpy_for_chumpy() -> None:
    """Let chumpy import on NumPy 2.

    CoMotion's SMPL loader imports chumpy, which still does
    ``from numpy import bool, int, float``.  Those names are gone in
    NumPy 2.  Setting them on the module is enough; the body model
    itself is unchanged.  NumPy warns on the lookup itself, so the
    warning filter covers the whole loop.
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)
        for old, new in _NUMPY_ALIASES.items():
            if not hasattr(np, old):
                setattr(np, old, getattr(np, new))


def _wrap_llama_decoder_layers(module: torch.nn.Module) -> None:
    """Restore the transformers 4.48 decoder return on transformers 5.

    ``LlamaDecoderLayer.forward`` used to return ``(hidden_states,)``.
    CoMotion's ``_call_llama`` keeps that tuple's first element.
    Transformers 5 returns the tensor directly, so ``[0]`` would keep
    only the first batch row.  Wrapping the layers this module owns
    puts the tuple back.  Eager attention matches the 4.48 default.
    """
    import transformers
    from mmengine.utils import digit_version

    if digit_version(transformers.__version__) < digit_version('5.0.0'):
        return
    for layer in module.modules():
        if layer.__class__.__name__ != 'LlamaDecoderLayer':
            continue
        attn = getattr(layer, 'self_attn', None)
        cfg = getattr(attn, 'config', None)
        if cfg is not None and getattr(cfg, '_attn_implementation',
                                        None) is None:
            cfg._attn_implementation = 'eager'
        original = layer.forward

        def forward(*args, _original=original, **kwargs):
            out = _original(*args, **kwargs)
            if isinstance(out, torch.Tensor):
                return (out, )
            return out

        layer.forward = forward


def _asset_paths(comotion_root: str) -> Tuple[str, str, str]:
    data = os.path.join(comotion_root, 'src', 'comotion_demo', 'data')
    return (
        os.path.join(data, 'comotion_detection_checkpoint.pt'),
        os.path.join(data, 'comotion_refine_checkpoint.pt'),
        os.path.join(data, 'smpl', 'SMPL_NEUTRAL.pkl'),
    )


def _ensure_comotion_on_path(comotion_root: str) -> str:
    """Put the checkout's ``src`` on ``sys.path`` and check the body model."""
    comotion_root = os.path.abspath(comotion_root)
    src = os.path.join(comotion_root, 'src')
    if not os.path.isdir(src):
        raise FileNotFoundError(
            f'CoMotion checkout not found at {comotion_root}. '
            'Add it with `git submodule update --init external/ml-comotion`.')
    detect_pt, refine_pt, smpl_pkl = _asset_paths(comotion_root)
    missing = [p for p in (detect_pt, refine_pt, smpl_pkl)
               if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            'CoMotion cannot run without its checkpoints and the neutral '
            'SMPL v1.1.0 body model. Missing:\n  '
            + '\n  '.join(missing)
            + '\nDownload the checkpoints with '
            'external/ml-comotion/get_pretrained_models.sh. '
            'Download basicmodel_neutral_lbs_10_207_0_v1.1.0.pkl from '
            'https://smpl.is.tue.mpg.de/ and copy it to '
            f'{smpl_pkl}.')
    if src not in sys.path:
        sys.path.insert(0, src)
    return comotion_root


def _load_comotion_modules(comotion_root: str):
    """Import upstream after the NumPy shim. Returns ``(comotion, dataloading, helper, track)``."""
    _shim_numpy_for_chumpy()
    _ensure_comotion_on_path(comotion_root)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=FutureWarning,
                                module='chumpy')
        from comotion_demo.models import comotion
        from comotion_demo.utils import dataloading, helper
        from comotion_demo.utils import track as track_utils
    return comotion, dataloading, helper, track_utils


def _keypoint_names(convention: str) -> List[str]:
    from mmpose.datasets.transforms.keypoint_registry import get_keypoints
    return list(get_keypoints(convention))


def posetrack17_from_coco(coco_xy: np.ndarray,
                          fill_head: bool = True
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Project COCO-17 pixel coordinates into PoseTrack-17.

    Joints the two layouts share are copied and given score 1.  Eyes have
    no PoseTrack counterpart and are dropped.  ``head_bottom`` and
    ``head_top`` stay at score 0 unless ``fill_head`` is set, in which
    case they are the train-fitted substitute described on
    :data:`_HEAD_FILL_A_BOTTOM`.  Their score is the minimum of the nose
    and shoulder scores (always 1 for a copied joint).

    Args:
        coco_xy: ``(N, 17, 2)`` COCO-17 coordinates.
        fill_head: Whether to synthesise the two head joints.

    Returns:
        Tuple of ``(N, 17, 2)`` coordinates and ``(N, 17)`` scores, in
        PoseTrack-17 order.
    """
    coco_xy = np.asarray(coco_xy, dtype=np.float32).reshape(-1, 17, 2)
    n = coco_xy.shape[0]
    coco_names = _keypoint_names('coco')
    pt_names = _keypoint_names('posetrack18')
    coco_index = {name: i for i, name in enumerate(coco_names)}
    kpts = np.zeros((n, len(pt_names), 2), dtype=np.float32)
    scores = np.zeros((n, len(pt_names)), dtype=np.float32)
    for dst, name in enumerate(pt_names):
        src = coco_index.get(name)
        if src is None:
            continue
        kpts[:, dst] = coco_xy[:, src]
        scores[:, dst] = 1.0
    if fill_head and n:
        name_at = {name: i for i, name in enumerate(pt_names)}
        nose = name_at['nose']
        left = name_at['left_shoulder']
        right = name_at['right_shoulder']
        bottom = name_at['head_bottom']
        top = name_at['head_top']
        nose_xy = kpts[:, nose]
        mid = 0.5 * (kpts[:, left] + kpts[:, right])
        delta = nose_xy - mid
        kpts[:, bottom] = mid + _HEAD_FILL_A_BOTTOM * delta
        kpts[:, top] = nose_xy + _HEAD_FILL_B_TOP * delta
        scores[:, bottom] = np.minimum(
            scores[:, nose], np.minimum(scores[:, left], scores[:, right]))
        scores[:, top] = scores[:, bottom]
    return kpts, scores


def _empty_instances(num_keypoints: int) -> InstanceData:
    pred = InstanceData()
    pred.keypoints = np.zeros((0, num_keypoints, 2), dtype=np.float32)
    pred.keypoint_scores = np.zeros((0, num_keypoints), dtype=np.float32)
    pred.keypoints_visible = np.zeros((0, num_keypoints), dtype=np.float32)
    pred.bboxes = np.zeros((0, 4), dtype=np.float32)
    pred.bbox_scores = np.zeros((0, ), dtype=np.float32)
    pred.track_ids = np.zeros((0, ), dtype=np.int32)
    return pred


@MODELS.register_module()
class CoMotionPoseEstimator(BaseModel):
    """MMPose wrapper around the unmodified CoMotion demo pipeline.

    Args:
        checkpoint (str | None): Must be
            ``comotion_detection_checkpoint.pt`` inside the checkout (or a
            symlink to it).  ``init_model`` writes the benchmark CLI path
            here.  The refine checkpoint is loaded from the sibling file
            by upstream.  ``None`` still loads those fixed paths.
        comotion_root (str): Path to the ``external/ml-comotion`` checkout.
        fill_head (bool): Synthesise ``head_bottom`` / ``head_top``.
            Default ``True``.  This is not part of CoMotion.
        min_matched_frames (int): Passed to ``cleanup_tracks``.  The demo
            uses 1.  Default 1.
        device (str): Accepted because ``init_model`` sets it.  Weights
            stay on CPU until ``.to(device)``.
        data_preprocessor (dict | None): Must leave pixels in ``0-255``
            and set ``bgr_to_rgb=True``.  CoMotion normalises internally
            and its shot detector expects RGB uint8.
        init_cfg: Unused.
    """

    emits_track_ids = True

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        comotion_root: str = _COMOTION_ROOT,
        fill_head: bool = True,
        min_matched_frames: int = 1,
        device: str = 'cuda:0',
        data_preprocessor: Optional[dict] = None,
        init_cfg=None,
    ):
        del device  # init_model moves the module after construction.
        if data_preprocessor is not None and isinstance(data_preprocessor,
                                                         dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        comotion_mod, dataloading, helper, track_utils = _load_comotion_modules(
            comotion_root)
        detect_pt, _, _ = _asset_paths(os.path.abspath(comotion_root))
        if checkpoint is not None and (
                os.path.realpath(checkpoint) != os.path.realpath(detect_pt)):
            raise ValueError(
                'CoMotion loads its own checkpoints from '
                f'{detect_pt} and the sibling refine file. '
                f'pose_checkpoint must be that detection file '
                f'(symlinks are fine). Got {checkpoint}.')

        self._dataloading = dataloading
        self._helper = helper
        self._track_utils = track_utils
        self.fill_head = bool(fill_head)
        self.min_matched_frames = int(min_matched_frames)
        self.cleanup_seconds = 0.0
        self._pt_names = _keypoint_names('posetrack18')
        self._needs_init = True
        self._pending: List = []
        self._detections: List = []
        self._tracks: List = []
        self._image_res: Optional[Tuple[int, int]] = None
        self._last_K: Optional[torch.Tensor] = None
        # Pretrained=True reads the two checkpoints inside CoMotion.__init__.
        self.comotion = comotion_mod.CoMotion(use_coreml=False, pretrained=True)
        _wrap_llama_decoder_layers(self.comotion)
        self.comotion.eval()

    @property
    def output_keypoint_names(self) -> List[str]:
        """PoseTrack-17 names, including the two synthesised head joints."""
        return list(self._pt_names)

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        if mode == 'loss':
            raise NotImplementedError(
                'CoMotionPoseEstimator is for inference only.')
        raise ValueError(f'Unsupported mode "{mode}"')

    def reset_tracking(self) -> None:
        """Drop online track state. Does not decode the buffered clip.

        ``finalize_sequence`` must run first, while the buffer still holds
        the clip that just ended.  The driver does that.
        """
        from scenedetect.detectors import ContentDetector

        self._needs_init = True
        self.comotion.frame_count = 0
        self.comotion.shot_detector = ContentDetector(
            threshold=50.0, min_scene_len=3)
        if hasattr(self.comotion, 'handler'):
            del self.comotion.handler

    def _as_uint8_rgb(self, inputs: torch.Tensor) -> torch.Tensor:
        prep = self.data_preprocessor
        if prep is not None and getattr(prep, '_enable_normalize', False):
            raise ValueError(
                'CoMotion normalises internally. Leave the '
                'PoseDataPreprocessor mean and std unset so pixels stay '
                'in 0-255.')
        if prep is not None and not getattr(prep, '_channel_conversion',
                                             False):
            raise ValueError(
                'LoadImage is BGR. Set data_preprocessor.bgr_to_rgb=True '
                'so CoMotion receives RGB.')
        if isinstance(inputs, (list, tuple)):
            if len(inputs) != 1:
                raise ValueError(
                    'CoMotion tracking accepts one frame per predict().')
            inputs = inputs[0]
        if inputs.dim() == 4:
            if inputs.shape[0] != 1:
                raise ValueError(
                    'CoMotion tracking accepts one frame per predict(), '
                    f'got a batch of {inputs.shape[0]}.')
            inputs = inputs[0]
        if inputs.dim() != 3 or inputs.shape[0] != 3:
            raise ValueError(
                f'Expected a CHW RGB frame, got shape {tuple(inputs.shape)}.')
        image = inputs.detach()
        if image.is_floating_point():
            image = image.clamp(0, 255).round().to(torch.uint8)
        elif image.dtype != torch.uint8:
            image = image.to(torch.uint8)
        return image.cpu()

    def predict(self, inputs: torch.Tensor,
                data_samples: SampleList) -> SampleList:
        """Run one frame of the demo loop and buffer it for cleanup.

        The returned sample has no instances.  ``finalize_sequence``
        replaces ``pred_instances`` once the whole clip has been seen.
        """
        if len(data_samples) != 1:
            raise ValueError(
                'CoMotion tracking accepts one frame per predict().')
        image = self._as_uint8_rgb(inputs)
        sample = data_samples[0]
        ori = sample.metainfo.get('ori_shape', None)
        height, width = int(image.shape[-2]), int(image.shape[-1])
        if ori is not None and (height, width) != (int(ori[0]), int(ori[1])):
            raise ValueError(
                f'CoMotion must see the original frame, got '
                f'{(height, width)} but ori_shape is '
                f'{(int(ori[0]), int(ori[1]))}. Remove any resize from '
                'the test pipeline.')

        K = self._dataloading.get_default_K(image)
        if self._needs_init or not hasattr(self.comotion, 'handler'):
            self.comotion.init_tracks((height, width))
            self._needs_init = False
            self._image_res = (height, width)
        elif self._image_res != (height, width):
            warnings.warn(
                'CoMotion demo uses the first frame\'s resolution and the '
                f'last frame\'s intrinsics for the whole clip. This clip '
                f'changed from {self._image_res} to {(height, width)}.',
                stacklevel=2)

        detection, track = self.comotion(image, K)
        self._detections.append(
            {k: v.detach().cpu() for k, v in detection.items()})
        self._tracks.append(track.detach().cpu())
        self._last_K = K.detach().cpu()
        sample.pred_instances = _empty_instances(len(self._pt_names))
        self._pending.append(sample)
        return data_samples

    def finalize_sequence(self) -> None:
        """Run the demo's ``cleanup_tracks`` and rewrite this clip's samples."""
        if not self._pending:
            return
        started = time.perf_counter()
        samples = self._pending
        detections = self._detections
        tracks = self._tracks
        image_res = self._image_res
        K = None if self._last_K is None else self._last_K.detach().cpu()
        self._pending = []
        self._detections = []
        self._tracks = []
        self._image_res = None
        self._last_K = None

        device = next(self.comotion.parameters()).device
        decoder = self.comotion.smpl_decoder
        decoder.cpu()
        try:
            self._decode_clip(samples, detections, tracks, image_res, K,
                              decoder)
        finally:
            decoder.to(device)
            self.cleanup_seconds += time.perf_counter() - started

    def _decode_clip(self, samples, detections, tracks, image_res, K,
                     decoder) -> None:
        if K is None or image_res is None or not detections:
            return
        det = {k: [d[k] for d in detections] for k in detections[0].keys()}
        stacked = torch.stack(tracks, 1)
        tr = {
            k: getattr(stacked, k)
            for k in ('id', 'pose', 'trans', 'betas')
        }
        track_ref = self._track_utils.cleanup_tracks(
            {'detections': det, 'tracks': tr},
            K,
            decoder,
            min_matched_frames=self.min_matched_frames,
        )
        by_frame = {}
        if track_ref:
            frame_idxs, track_idxs = self._track_utils.convert_to_idxs(
                track_ref, tr['id'][0].squeeze(-1).long())
            preds = {
                k: v[0, frame_idxs, track_idxs]
                for k, v in tr.items()
            }
            ids = preds['id'].squeeze(-1).long()
            smpl = {k: preds[k] for k in ('betas', 'pose', 'trans')}
            boxes = self._track_utils.bboxes_from_smpl(
                decoder, smpl, image_res, K).reshape(-1, 4)
            coco3d = decoder(
                smpl['betas'], smpl['pose'], smpl['trans'],
                output_format='joints_coco')
            coco2d = self._helper.project_to_2d(K, coco3d).detach().cpu().numpy()
            boxes_np = boxes.detach().cpu().numpy().astype(np.float32)
            ids_np = ids.detach().cpu().numpy().astype(np.int32)
            kpts, scores = posetrack17_from_coco(
                coco2d, fill_head=self.fill_head)
            for row, frame_idx in enumerate(frame_idxs.tolist()):
                by_frame.setdefault(int(frame_idx), []).append(row)

            for frame_idx, rows in by_frame.items():
                if frame_idx < 0 or frame_idx >= len(samples):
                    raise RuntimeError(
                        f'cleanup_tracks returned frame {frame_idx} for a '
                        f'clip of {len(samples)} frames.')
                chosen = np.asarray(rows, dtype=np.int64)
                self._write_frame(
                    samples[frame_idx],
                    kpts[chosen],
                    scores[chosen],
                    boxes_np[chosen],
                    ids_np[chosen],
                )

    def _write_frame(self, sample, kpts, scores, boxes, track_ids) -> None:
        pred = InstanceData()
        pred.keypoints = np.asarray(kpts, dtype=np.float32)
        pred.keypoint_scores = np.asarray(scores, dtype=np.float32)
        pred.keypoints_visible = np.asarray(scores, dtype=np.float32)
        pred.bboxes = np.asarray(boxes, dtype=np.float32)
        n = pred.keypoints.shape[0]
        pred.bbox_scores = np.ones((n, ), dtype=np.float32)
        pred.track_ids = np.asarray(track_ids, dtype=np.int32).reshape(-1)
        sample.pred_instances = pred
