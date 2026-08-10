"""Inference-only wrapper for QueryPose-light (``external/QueryPose``).

QueryPose (`Xiao et al., NeurIPS 2022 <https://arxiv.org/abs/2212.07855>`_) is
a Detectron2-based sparse end-to-end multi-person pose estimator.  Only the
**light** HRNet-W32 / W48 checkpoints are publicly released (README AP 69.8 /
71.0).  Paper Table numbers that require ``LIGHT_VERSION=False`` or Swin-L /
ResNet50 weights are **not** reproducible from public artifacts — do not
report those under this wrapper's name.

This wrapper loads the upstream QueryPose + vendored Detectron2 unmodified
(after ``python setup.py build develop`` in ``external/QueryPose``) and
translates ``Instances`` to MMPose ``PoseDataSample``.

Native boxes and scores are used (not derived).

**Fidelity note:** upstream pose head returns keypoints as ``(B*N, K, 3)``
while ``QueryPose.inference`` expects ``(B, N, K, 3)``.  The wrapper reshapes
before calling inference (no submodule edit).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import numpy as np
import torch
from mmengine.model import BaseModel
from mmengine.structures import InstanceData

from mmpose.registry import MODELS
from mmpose.utils.typing import SampleList

_QUERYPOSE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'external',
                 'QueryPose'))


def _ensure_querypose_importable(querypose_root: str) -> None:
    """Put vendored Detectron2 + projects/querypose ahead of any pip D2."""
    # Pillow>=10 dropped Image.LINEAR (alias of BILINEAR); Detectron2 0.3 still
    # references it at import time.
    from PIL import Image
    if not hasattr(Image, 'LINEAR'):
        Image.LINEAR = Image.BILINEAR  # type: ignore[attr-defined]

    qp_root = os.path.abspath(querypose_root)
    project_root = os.path.join(qp_root, 'projects', 'querypose')
    for path in (project_root, qp_root):
        if path not in sys.path:
            sys.path.insert(0, path)
    try:
        import detectron2  # noqa: F401
        from detectron2 import _C  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            'QueryPose requires the vendored Detectron2 CUDA extension. '
            'Build it with:\n'
            '  cd external/QueryPose && python setup.py build develop\n'
            'Use an isolated env — this package name clashes with pip '
            f'detectron2. Original error: {exc}') from exc


@MODELS.register_module()
class QueryPosePoseEstimator(BaseModel):
    """MMPose inference wrapper for QueryPose-light (bottom-up, end-to-end).

    Args:
        config_file (str): Upstream yaml under ``projects/querypose/configs/``.
        checkpoint (str | None): Local light ``.pth`` path.
        querypose_root (str): Path to ``external/QueryPose``.
        score_thr (float): Filter instance scores; ``0.0`` keeps all proposals.
        device (str): Device string for Detectron2 ``MODEL.DEVICE``.
        data_preprocessor (dict | None): Should leave RGB float in ~[0, 255]
            (mean 0, std 1, ``bgr_to_rgb=True``) so QueryPose's own normalizer
            matches DefaultPredictor.
        init_cfg: Unused; API compatibility.
    """

    def __init__(
        self,
        config_file: str = (
            'projects/querypose/configs/querypose.hrnet32.100pro.3x.yaml'),
        checkpoint: Optional[str] = None,
        querypose_root: str = _QUERYPOSE_ROOT,
        score_thr: float = 0.0,
        device: str = 'cuda:0',
        data_preprocessor: Optional[dict] = None,
        init_cfg=None,
    ):
        if data_preprocessor is not None and isinstance(data_preprocessor,
                                                         dict):
            data_preprocessor = MODELS.build(data_preprocessor)
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.score_thr = float(score_thr)
        self.num_keypoints = 17

        querypose_root = os.path.abspath(querypose_root)
        _ensure_querypose_importable(querypose_root)

        from detectron2.checkpoint import DetectionCheckpointer
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model
        from querypose import add_querypose_config  # noqa: WPS433

        cfg_path = config_file
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(querypose_root, cfg_path)
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f'QueryPose config not found: {cfg_path}')

        cfg = get_cfg()
        add_querypose_config(cfg)
        cfg.merge_from_file(cfg_path)
        # Released HRNet configs already set LIGHT_VERSION=True; keep explicit.
        cfg.MODEL.QueryPose.LIGHT_VERSION = True
        cfg.MODEL.DEVICE = device if torch.cuda.is_available() else 'cpu'
        if checkpoint is None:
            raise FileNotFoundError(
                'QueryPosePoseEstimator requires a local light checkpoint. '
                'Download querypose_hrnet32/48 from the README Google Drive '
                'links into data/models/.')
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(
                f'QueryPose checkpoint not found: {checkpoint}. '
                'Only light HRNet32/48 weights are published '
                '(README AP 69.8 / 71.0).')
        cfg.MODEL.WEIGHTS = checkpoint
        cfg.freeze()

        model = build_model(cfg)
        DetectionCheckpointer(model).load(checkpoint)
        model.eval()

        # Upstream pose head returns keypoints as (B*N, K, 3) while class/box
        # heads return (B, N, ...). QueryPose.inference zips over batch and
        # expects (B, N, K, 3). Reshape here without editing the submodule.
        _orig_inference = model.inference

        def _inference_with_batch_dim(box_cls, box_pred, kps_pred, image_sizes,
                                      _self=model, _orig=_orig_inference):
            bsz, num_prop = box_cls.shape[:2]
            if (kps_pred.dim() == 3
                    and kps_pred.shape[0] == bsz * num_prop
                    and kps_pred.shape[1] == _self.num_kps):
                kps_pred = kps_pred.view(bsz, num_prop, _self.num_kps, -1)
            return _orig(box_cls, box_pred, kps_pred, image_sizes)

        model.inference = _inference_with_batch_dim  # type: ignore[method-assign]

        self.querypose_model = model
        self.cfg = cfg

    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'predict':
            return self.predict(inputs, data_samples)
        if mode == 'loss':
            raise NotImplementedError(
                'QueryPosePoseEstimator is for inference only.')
        raise ValueError(f'Unsupported mode "{mode}"')

    def _inference_forward(self, batched_inputs: list):
        """NN forward (FPS metric target)."""
        return self.querypose_model(batched_inputs, do_postprocess=True)

    def predict(self, inputs: torch.Tensor,
                data_samples: SampleList) -> SampleList:
        """Run QueryPose-light and pack native boxes/keypoints."""
        batched_inputs = []
        for i, ds in enumerate(data_samples):
            ori = ds.metainfo.get('ori_shape', None)
            if ori is None:
                h, w = int(inputs.shape[-2]), int(inputs.shape[-1])
            else:
                h, w = int(ori[0]), int(ori[1])
            batched_inputs.append({
                'image': inputs[i].contiguous(),
                'height': h,
                'width': w,
            })

        with torch.no_grad():
            outputs = self._inference_forward(batched_inputs)

        k = self.num_keypoints
        for i, out in enumerate(outputs):
            inst = out['instances']
            if len(inst) == 0:
                kpts = np.zeros((0, k, 2), dtype=np.float32)
                kpt_scores = np.zeros((0, k), dtype=np.float32)
                bboxes = np.zeros((0, 4), dtype=np.float32)
                bbox_scores = np.zeros((0, ), dtype=np.float32)
            else:
                scores = inst.scores.detach().float().cpu().numpy()
                boxes = inst.pred_boxes.tensor.detach().float().cpu().numpy()
                kps = inst.pred_keypoints.detach().float().cpu().numpy()
                keep = scores > self.score_thr
                scores = scores[keep]
                boxes = boxes[keep]
                kps = kps[keep]
                n = int(scores.shape[0])
                if n == 0:
                    kpts = np.zeros((0, k, 2), dtype=np.float32)
                    kpt_scores = np.zeros((0, k), dtype=np.float32)
                    bboxes = np.zeros((0, 4), dtype=np.float32)
                    bbox_scores = np.zeros((0, ), dtype=np.float32)
                else:
                    kpts = kps[:, :, :2].astype(np.float32)
                    kpt_scores = kps[:, :, 2].astype(np.float32)
                    bboxes = boxes[:, :4].astype(np.float32)
                    bbox_scores = scores.astype(np.float32)

            pred = InstanceData()
            pred.keypoints = kpts
            pred.keypoint_scores = kpt_scores
            pred.keypoints_visible = kpt_scores
            pred.bboxes = bboxes
            pred.bbox_scores = bbox_scores
            data_samples[i].pred_instances = pred

        return data_samples
