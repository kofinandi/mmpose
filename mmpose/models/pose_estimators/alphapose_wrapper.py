"""Wrapper integrating AlphaPose (pose + re-ID tracking) from
``external/AlphaPose`` into MMPose v1.x for evaluation.

AlphaPose (`Fang et al., TPAMI 2022
<https://ieeexplore.ieee.org/document/9954214>`_) is a **top-down** pose
estimator with an online multi-person tracker.  This wrapper reproduces
the released ``--pose_track`` pipeline
(``external/AlphaPose/scripts/demo_inference.py``), which per frame:

1. crops each detection to 256x192 and runs the SPPE (FastPose / HRNet),
2. runs a **separate re-ID network over those same crops**, and
3. associates the resulting embeddings against the active tracks with a
   Kalman motion model and a JDE-style three-stage cascade.

.. note:: **What "re-ID" means here.**  The re-ID features are *not* taken
   from the pose backbone.  ``trackers/tracker_api.py:231`` runs
   ``feats = self.model(inps)`` where ``self.model`` is an **OSNet-AIN**
   (``osnet_ain_x1_0``) trained on MSMT17, and ``inps`` is the same
   preprocessed crop batch the pose network receives -- shared *input*,
   not shared features.  A pose-backbone-derived variant (``ResModel``)
   exists in the repo but is commented out at ``tracker_api.py:199`` and
   has no released weights, so it is not what the paper's numbers came
   from and is not implemented here.

The upstream association algorithm is used **verbatim**: this wrapper
subclasses ``trackers.tracker_api.Tracker`` and overrides only
``__init__`` (to accept an already-constructed re-ID module instead of
building one on a fixed device, so MMEngine's ``.to(device)`` owns
placement).  ``Tracker.update()`` -- embedding distance fused with Kalman
motion gating at 0.7, IoU fallback at 0.5, unconfirmed-track IoU at 0.7,
``track_buffer`` ageing -- is inherited unmodified.

Interface adaptations, all confined to this file:

- **Preprocessing** is MMPose's ``GetBBoxCenterScale(padding=1.25)`` +
  ``TopdownAffine``, which is the same math as upstream's
  ``SimpleTransform.test_transform`` (``_box_to_center_scale`` with
  ``scale_mult=1.25`` then ``get_affine_transform``; the aspect-ratio fix
  and the 1.25 scaling commute).  Normalisation matches upstream exactly:
  ``img/255`` then subtract ``(0.406, 0.457, 0.480)`` per channel with no
  division by std, on **RGB** input (upstream reads frames as RGB --
  ``alphapose/utils/detector.py:162,217`` -- and subtracts those constants
  from channels 0,1,2 in that order).  Expressed as a
  ``PoseDataPreprocessor`` this is ``mean=[103.53, 116.535, 122.4]``,
  ``std=[255., 255., 255.]``, ``bgr_to_rgb=True``.
- **Decoding** calls upstream's own
  ``alphapose.utils.transforms.get_func_heatmap_to_coord(cfg)`` on the
  upstream crop box, i.e. the released argmax + quarter-pixel-offset
  decode, not an MMPose codec.
- **Pose NMS is skipped**, matching upstream: ``writer.py`` runs
  ``pose_nms`` only ``if not self.opt.pose_track``.
- **Checkpoint loading**: the released SPPE ``.pth`` files are flat state
  dicts, so a ``load_state_dict`` pre-hook adds the ``pose_model.``
  prefix and plain ``init_model(config, checkpoint)`` works.  The re-ID
  weights are a *second* file, loaded in ``__init__`` from
  ``tracker.loadmodel`` via upstream's own ``load_pretrained_weights``.

Output shape differs from upstream's JSON writer in two documented ways
(neither touches the association, which is what the paper contributes):

- **Boxes are the detector's boxes**, not ``STrack.tlbr`` (the Kalman-
  smoothed box upstream emits).  Keypoints are identical either way --
  upstream also decodes them from the *detection's* crop box, not the
  Kalman box -- and reporting detector boxes keeps this model's bbox
  output comparable with every other model in ``tools/benchmark_e2e.py``.
- **Instances the tracker did not return keep ``track_id = -1``** instead
  of being dropped, so the pose predictions stay aligned one-to-one with
  the detections the driver supplied.

Usage example (in a config file)::

    emits_track_ids = True   # routes tools/benchmark_e2e.py to run_tracking

    model = dict(
        type='AlphaPosePoseEstimator',
        alphapose_root='external/AlphaPose',
        alphapose_cfg='configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml',
        tracker=dict(
            arch='osnet_ain',
            loadmodel='data/models/alphapose/osnet_ain_x1_0_msmt17_...pth',
        ),
        data_preprocessor=dict(
            type='PoseDataPreprocessor',
            mean=[103.53, 116.535, 122.4],
            std=[255.0, 255.0, 255.0],
            bgr_to_rgb=True,
        ),
    )
"""

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmpose.models.pose_estimators.alphapose_compat import (
    ensure_alphapose_on_path, install_alphapose_shims)
from mmpose.registry import MODELS
from mmpose.utils.typing import SampleList

_ALPHAPOSE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'external',
                 'AlphaPose'))

# Upstream defaults from external/AlphaPose/trackers/tracker_cfg.py.  Only
# `loadmodel` has no sensible default (it points into the repo's own
# trackers/weights/ dir, whereas this repo keeps checkpoints under
# data/models/).
_DEFAULT_TRACKER_CFG = dict(
    nid=1000,
    arch='osnet_ain',
    loadmodel=None,
    frame_rate=30,
    track_buffer=240,
    conf_thres=0.5,
    nms_thres=0.4,
    iou_thres=0.5,
)


def _build_upstream_tracker_class():
    """Return a ``Tracker`` subclass that takes a prebuilt re-ID module.

    Upstream ``Tracker.__init__`` constructs the re-ID network, wraps it in
    ``nn.DataParallel(device_ids=args.gpus)`` and moves it to a device --
    all at construction time, before MMEngine has decided where the model
    lives.  Only that construction is replaced; ``update()``, which is the
    published association algorithm, is inherited untouched.
    """
    from trackers.tracker_api import Tracker
    from trackers.utils.kalman_filter import KalmanFilter

    class _AlphaPoseTracker(Tracker):

        def __init__(self, opt, reid_model):
            self.opt = opt
            self.num_joints = 17
            self.frame_rate = opt.frame_rate
            self.model = reid_model

            self.tracked_stracks = []
            self.lost_stracks = []
            self.removed_stracks = []

            self.frame_id = 0
            self.det_thresh = opt.conf_thres
            self.buffer_size = int(self.frame_rate / 30.0 * opt.track_buffer)
            self.max_time_lost = self.buffer_size

            self.kalman_filter = KalmanFilter()

    return _AlphaPoseTracker


@MODELS.register_module()
class AlphaPosePoseEstimator(BaseModel):
    """MMPose v1.x wrapper for AlphaPose with its re-ID pose tracker.

    Args:
        alphapose_cfg (str): Path to an upstream AlphaPose YAML config
            (e.g. ``configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml``),
            relative to ``alphapose_root`` unless absolute.  Defines the
            SPPE architecture, input/heatmap size and decode function.
        alphapose_root (str): Path to the ``external/AlphaPose`` checkout.
        tracker (dict, optional): Overrides for the upstream tracker config
            (``external/AlphaPose/trackers/tracker_cfg.py``).  ``loadmodel``
            (path to the OSNet-AIN re-ID weights) is required.  Pass
            ``None`` to disable tracking entirely and run pose only, in
            which case every ``track_id`` is ``-1``.
        data_preprocessor (dict, optional): Built with ``MODELS.build``.
            Must reproduce upstream's normalisation; see the module
            docstring.
        init_cfg (dict, optional): MMEngine init config.
    """

    def __init__(
        self,
        alphapose_cfg: str,
        alphapose_root: str = _ALPHAPOSE_ROOT,
        tracker: Optional[dict] = None,
        data_preprocessor: Optional[dict] = None,
        init_cfg: Optional[dict] = None,
    ):
        if data_preprocessor is not None and isinstance(data_preprocessor,
                                                        dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        install_alphapose_shims()
        root = ensure_alphapose_on_path(alphapose_root)

        from alphapose.models import builder
        from alphapose.utils.config import update_config
        from alphapose.utils.transforms import get_func_heatmap_to_coord

        cfg_path = (alphapose_cfg if os.path.isabs(alphapose_cfg) else
                    os.path.join(root, alphapose_cfg))
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f'AlphaPose config not found: {cfg_path!r}. It should be a '
                f'path relative to {root!r} (or absolute).')
        self.ap_cfg = update_config(cfg_path)

        # Built with random init; the released checkpoint lands on top via
        # MMEngine's load_checkpoint (see _remap_checkpoint_keys).
        self.ap_cfg.MODEL.PRETRAINED = ''
        self.ap_cfg.MODEL.TRY_LOAD = ''
        self.pose_model = builder.build_sppe(
            self.ap_cfg.MODEL, preset_cfg=self.ap_cfg.DATA_PRESET)
        self.pose_model.eval()

        self._heatmap_to_coord = get_func_heatmap_to_coord(self.ap_cfg)
        if isinstance(self._heatmap_to_coord, list):
            raise NotImplementedError(
                f'{cfg_path!r} uses LOSS.TYPE=Combined, whose decode splits '
                f'body/foot from face/hand keypoints. Only the single-decode '
                f'configs (MSELoss / L1JointRegression) are wired up here.')
        self._hm_size = list(self.ap_cfg.DATA_PRESET.HEATMAP_SIZE)
        self._norm_type = self.ap_cfg.LOSS.get('NORM_TYPE', None)
        self.num_joints = int(self.ap_cfg.DATA_PRESET.NUM_JOINTS)

        self.reid_model = None
        self._tracker_opt = None
        self.tracker = None
        if tracker is not None:
            self._init_tracker(tracker)

        self._register_load_state_dict_pre_hook(self._remap_checkpoint_keys)

    # ── Construction helpers ───────────────────────────────────────────

    def _init_tracker(self, tracker: dict) -> None:
        """Build the OSNet-AIN re-ID net and the upstream tracker."""
        from easydict import EasyDict as edict

        from trackers.ReidModels.osnet import osnet_x1_0
        from trackers.ReidModels.osnet_ain import osnet_ain_x1_0
        from trackers.ReidModels.resnet_fc import resnet50_fc512
        from trackers.utils.utils import load_pretrained_weights

        opt = edict(dict(_DEFAULT_TRACKER_CFG, **tracker))
        if not opt.loadmodel:
            raise ValueError(
                'tracker.loadmodel is required: it is the path to the '
                'OSNet-AIN re-ID weights AlphaPose\'s --pose_track uses '
                '(osnet_ain_x1_0_msmt17_...pth, see '
                'external/AlphaPose/trackers/README.md). Pass tracker=None '
                'to run pose estimation without tracking instead.')
        if not os.path.isfile(opt.loadmodel):
            raise FileNotFoundError(
                f'Re-ID weights not found: {opt.loadmodel!r}. Download the '
                f'"human reid model" linked from '
                f'external/AlphaPose/trackers/README.md and place it there.')

        builders = {
            'osnet_ain': osnet_ain_x1_0,
            'osnet': osnet_x1_0,
            'res50-fc512': resnet50_fc512,
        }
        if opt.arch not in builders:
            raise ValueError(
                f'Unknown tracker.arch {opt.arch!r}; upstream '
                f'trackers/tracker_api.py supports {sorted(builders)}.')
        reid = builders[opt.arch](num_classes=1, pretrained=False)

        # Upstream loads weights into an nn.DataParallel, and
        # load_pretrained_weights matches keys by prepending 'module.'.
        # Wrap only for the load so the key mapping is upstream's, then
        # keep the bare module: DataParallel would pin the re-ID net to a
        # device chosen here rather than the one MMEngine moves us to.
        load_pretrained_weights(nn.DataParallel(reid), opt.loadmodel)
        reid.eval()

        self.reid_model = reid
        self._tracker_opt = opt
        self.tracker = _build_upstream_tracker_class()(opt, reid)

    def _remap_checkpoint_keys(self, state_dict, prefix, *args, **kwargs):
        """Re-prefix a flat upstream SPPE state dict onto ``pose_model.``.

        The released AlphaPose checkpoints (``fast_res50_256x192.pth`` etc.)
        are plain ``{layer: tensor}`` dicts for the SPPE alone, with no
        ``state_dict``/``meta`` wrapper.  Keys already belonging to this
        wrapper's own submodules are left alone so the re-ID weights loaded
        in ``__init__`` are not clobbered.

        The re-ID weights come from a *second* file (``tracker.loadmodel``)
        and are therefore absent from this checkpoint.  They are seeded back
        in from the already-loaded module so MMEngine does not report every
        ``reid_model.*`` parameter as missing -- a warning that would be
        pure noise, and would hide a real missing-key problem if one ever
        appeared.
        """
        target_prefix = prefix + 'pose_model.'
        skip_prefixes = (
            target_prefix,
            prefix + 'reid_model.',
            prefix + 'data_preprocessor.',
        )
        for k in list(state_dict.keys()):
            if k.startswith(prefix) and not any(
                    k.startswith(p) for p in skip_prefixes):
                state_dict[target_prefix + k[len(prefix):]] = state_dict.pop(k)

        if self.reid_model is not None:
            reid_prefix = prefix + 'reid_model.'
            for k, v in self.reid_model.state_dict().items():
                state_dict.setdefault(reid_prefix + k, v)

    # ── Tracking state ─────────────────────────────────────────────────

    def reset_tracking(self) -> None:
        """Clear all tracker state; called at every sequence boundary.

        Also resets the process-global ``BaseTrack._count`` so IDs restart
        from 1 in each video, matching what a fresh upstream process (which
        handles exactly one video) would produce.
        """
        if self.tracker is None:
            return
        from trackers.tracking.basetrack import BaseTrack

        BaseTrack._count = 0
        self.tracker.tracked_stracks = []
        self.tracker.lost_stracks = []
        self.tracker.removed_stracks = []
        self.tracker.frame_id = 0

    # ── Inference ──────────────────────────────────────────────────────

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        if mode == 'tensor':
            return self._inference_forward(inputs)
        raise NotImplementedError(
            f'{type(self).__name__} is inference-only; mode={mode!r} is not '
            f'supported.')

    def _inference_forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """SPPE forward only.

        Kept separate from :meth:`predict` so the ``FPS`` metric can time
        the network without the tracker and the InstanceData packing, as
        the other wrappers in this package do.
        """
        return self.pose_model(inputs)

    def predict(self, inputs: torch.Tensor,
                data_samples: SampleList) -> SampleList:
        """Estimate poses for one frame's detections and associate them.

        ``inputs`` holds every detection of a *single* frame -- see
        ``run_tracking`` in ``tools/benchmark_e2e.py``, which is what makes
        the tracker's per-frame contract satisfiable.
        """
        heatmaps = self._inference_forward(inputs)
        n = heatmaps.shape[0]

        # Upstream crop box: the padded, aspect-corrected box the affine
        # warp was built from, i.e. _center_scale_to_box(center, scale).
        crop_boxes = np.zeros((n, 4), dtype=np.float32)
        det_bboxes = np.zeros((n, 4), dtype=np.float32)
        det_scores = np.ones(n, dtype=np.float32)
        for i, ds in enumerate(data_samples):
            center = np.asarray(ds.metainfo['input_center']).reshape(2)
            scale = np.asarray(ds.metainfo['input_scale']).reshape(2)
            crop_boxes[i] = [
                center[0] - scale[0] * 0.5,
                center[1] - scale[1] * 0.5,
                center[0] + scale[0] * 0.5,
                center[1] + scale[1] * 0.5,
            ]
            gt = ds.gt_instances
            det_bboxes[i] = np.asarray(gt.bboxes).reshape(-1, 4)[0]
            if 'bbox_scores' in gt:
                det_scores[i] = float(np.asarray(gt.bbox_scores).reshape(-1)[0])

        hm_np = heatmaps.detach().cpu().numpy()
        track_ids = self._associate(inputs, hm_np, det_bboxes, det_scores)

        keypoints = np.zeros((n, self.num_joints, 2), dtype=np.float32)
        keypoint_scores = np.zeros((n, self.num_joints), dtype=np.float32)
        for i in range(n):
            coord, score = self._heatmap_to_coord(
                hm_np[i],
                crop_boxes[i].tolist(),
                hm_shape=self._hm_size,
                norm_type=self._norm_type)
            keypoints[i] = np.asarray(coord, dtype=np.float32)
            keypoint_scores[i] = np.asarray(
                score, dtype=np.float32).reshape(-1)

        for i, ds in enumerate(data_samples):
            pred = InstanceData()
            pred.keypoints = keypoints[i:i + 1]
            pred.keypoint_scores = keypoint_scores[i:i + 1]
            pred.keypoints_visible = (keypoint_scores[i:i + 1] > 0).astype(
                np.float32)
            pred.bboxes = det_bboxes[i:i + 1]
            pred.bbox_scores = det_scores[i:i + 1]
            pred.track_ids = track_ids[i:i + 1]
            ds.pred_instances = pred

        return data_samples

    def _associate(
        self,
        inputs: torch.Tensor,
        heatmaps: np.ndarray,
        det_bboxes: np.ndarray,
        det_scores: np.ndarray,
    ) -> np.ndarray:
        """Run the upstream tracker and map its output back to detections.

        Returns ``(N,)`` int32 track IDs aligned with the input detections;
        ``-1`` for a detection the tracker did not return.
        """
        n = len(det_bboxes)
        track_ids = np.full(n, -1, dtype=np.int32)
        if self.tracker is None:
            return track_ids

        # `cropped_boxes` is stored verbatim on each STrack and never read
        # by the association code (only echoed back by trackers/__init__.py
        # ::track), so it is used here as the carrier that maps a returned
        # track back to the detection index it matched.  Without it the
        # tracker's output order and the input order are unrelated, since
        # update() returns self.tracked_stracks.
        det_index_carrier = list(range(n))

        targets = self.tracker.update(
            # `img0` -- unused by Tracker.update (the re-ID features come
            # from `inps`, the pose crops, not the source frame), so the
            # wrapper never needs the full-resolution image.
            None,
            inputs,
            det_bboxes,
            heatmaps,
            det_index_carrier,
            '',
            det_scores,
            _debug=False,
        )
        for t in targets:
            idx = t.crop_box
            if isinstance(idx, (int, np.integer)) and 0 <= idx < n:
                track_ids[idx] = int(t.track_id)
        return track_ids
