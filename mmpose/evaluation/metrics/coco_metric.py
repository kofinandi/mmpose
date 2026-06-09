# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
from typing import Dict, Optional, Sequence

import numpy as np
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump, get_local_path, load
from mmengine.logging import MessageHub, MMLogger, print_log
from xtcocotools.coco import COCO
from xtcocotools.cocoeval import COCOeval

from mmpose.registry import METRICS
from mmpose.structures.bbox import bbox_xyxy2xywh
from ..functional import (oks_nms, soft_oks_nms, transform_ann, transform_pred,
                          transform_sigmas)


@METRICS.register_module()
class CocoMetric(BaseMetric):
    """COCO pose estimation task evaluation metric.

    Evaluate AR, AP, and mAP for keypoint detection tasks. Support COCO
    dataset and other datasets in COCO format. Please refer to
    `COCO keypoint evaluation <https://cocodataset.org/#keypoints-eval>`__
    for more details.

    Args:
        ann_file (str, optional): Path to the coco format annotation file.
            If not specified, ground truth annotations from the dataset will
            be converted to coco format. Defaults to None
        use_area (bool): Whether to use ``'area'`` message in the annotations.
            If the ground truth annotations (e.g. CrowdPose, AIC) do not have
            the field ``'area'``, please set ``use_area=False``.
            Defaults to ``True``
        iou_type (str): The same parameter as `iouType` in
            :class:`xtcocotools.COCOeval`, which can be ``'keypoints'``, or
            ``'keypoints_crowd'`` (used in CrowdPose dataset).
            Defaults to ``'keypoints'``
        score_mode (str): The mode to score the prediction results which
            should be one of the following options:

                - ``'bbox'``: Take the score of bbox as the score of the
                    prediction results.
                - ``'bbox_keypoint'``: Use keypoint score to rescore the
                    prediction results.
                - ``'bbox_rle'``: Use rle_score to rescore the
                    prediction results.

            Defaults to ``'bbox_keypoint'`
        keypoint_score_thr (float): The threshold of keypoint score. The
            keypoints with score lower than it will not be included to
            rescore the prediction results. Valid only when ``score_mode`` is
            ``bbox_keypoint``. Defaults to ``0.2``
        nms_mode (str): The mode to perform Non-Maximum Suppression (NMS),
            which should be one of the following options:

                - ``'oks_nms'``: Use Object Keypoint Similarity (OKS) to
                    perform NMS.
                - ``'soft_oks_nms'``: Use Object Keypoint Similarity (OKS)
                    to perform soft NMS.
                - ``'none'``: Do not perform NMS. Typically for bottomup mode
                    output.

            Defaults to ``'oks_nms'`
        nms_thr (float): The Object Keypoint Similarity (OKS) threshold
            used in NMS when ``nms_mode`` is ``'oks_nms'`` or
            ``'soft_oks_nms'``. Will retain the prediction results with OKS
            lower than ``nms_thr``. Defaults to ``0.9``
        format_only (bool): Whether only format the output results without
            doing quantitative evaluation. This is designed for the need of
            test submission when the ground truth annotations are absent. If
            set to ``True``, ``outfile_prefix`` should specify the path to
            store the output results. Defaults to ``False``
        pred_converter (dict, optional): Config dictionary for the prediction
            converter. The dictionary has the same parameters as
            'KeypointConverter'. Defaults to None.
        gt_converter (dict, optional): Config dictionary for the ground truth
            converter. The dictionary has the same parameters as
            'KeypointConverter'. Defaults to None.
        outfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., ``'a/b/prefix'``.
            If not specified, a temp file will be created. Defaults to ``None``
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be ``'cpu'`` or
            ``'gpu'``. Defaults to ``'cpu'``
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, ``self.default_prefix``
            will be used instead. Defaults to ``None``
    """
    default_prefix: Optional[str] = 'coco'

    # Default COCO sigmas used when gt_from_samples=True and dataset_meta
    # sigmas don't match the actual prediction keypoint count (e.g. MPII
    # dataset paired with a COCO-trained model via KeypointConverter).
    _COCO_SIGMAS = np.array([
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
        0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
    ])

    def __init__(self,
                 ann_file: Optional[str] = None,
                 use_area: bool = True,
                 iou_type: str = 'keypoints',
                 score_mode: str = 'bbox_keypoint',
                 keypoint_score_thr: float = 0.2,
                 nms_mode: str = 'oks_nms',
                 nms_thr: float = 0.9,
                 format_only: bool = False,
                 pred_converter: Dict = None,
                 gt_converter: Dict = None,
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 gt_from_samples: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.gt_from_samples = gt_from_samples
        # initialize coco helper with the annotation json file
        # if ann_file is not specified, initialize with the converted dataset
        if ann_file is not None:
            with get_local_path(ann_file) as local_path:
                self.coco = COCO(local_path)
        else:
            self.coco = None

        self.use_area = use_area
        self.iou_type = iou_type

        allowed_score_modes = ['bbox', 'bbox_keypoint', 'bbox_rle', 'keypoint']
        if score_mode not in allowed_score_modes:
            raise ValueError(
                "`score_mode` should be one of 'bbox', 'bbox_keypoint', "
                f"'bbox_rle', but got {score_mode}")
        self.score_mode = score_mode
        self.keypoint_score_thr = keypoint_score_thr

        allowed_nms_modes = ['oks_nms', 'soft_oks_nms', 'none']
        if nms_mode not in allowed_nms_modes:
            raise ValueError(
                "`nms_mode` should be one of 'oks_nms', 'soft_oks_nms', "
                f"'none', but got {nms_mode}")
        self.nms_mode = nms_mode
        self.nms_thr = nms_thr

        if format_only:
            assert outfile_prefix is not None, '`outfile_prefix` can not be '\
                'None when `format_only` is True, otherwise the result file '\
                'will be saved to a temp directory which will be cleaned up '\
                'in the end.'
        elif ann_file is not None:
            # do evaluation only if the ground truth annotations exist
            assert 'annotations' in load(ann_file), \
                'Ground truth annotations are required for evaluation '\
                'when `format_only` is False.'

        self.format_only = format_only
        self.outfile_prefix = outfile_prefix
        self.pred_converter = pred_converter
        self.gt_converter = gt_converter

    @property
    def dataset_meta(self) -> Optional[dict]:
        """Optional[dict]: Meta info of the dataset."""
        return self._dataset_meta

    @dataset_meta.setter
    def dataset_meta(self, dataset_meta: dict) -> None:
        """Set the dataset meta info to the metric."""
        if self.gt_converter is not None:
            dataset_meta['sigmas'] = transform_sigmas(
                dataset_meta['sigmas'], self.gt_converter['num_keypoints'],
                self.gt_converter['mapping'])
            dataset_meta['num_keypoints'] = len(dataset_meta['sigmas'])
        self._dataset_meta = dataset_meta

        if self.coco is None and not self.gt_from_samples:
            message = MessageHub.get_current_instance()
            ann_file = message.get_info(
                f"{dataset_meta['dataset_name']}_ann_file", None)
            if ann_file is not None:
                with get_local_path(ann_file) as local_path:
                    self.coco = COCO(local_path)
                print_log(
                    f'CocoMetric for dataset '
                    f"{dataset_meta['dataset_name']} has successfully "
                    f'loaded the annotation file from {ann_file}', 'current')

    def process(self, data_batch: Sequence[dict],
                data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from
                the model, each of which has the following keys:

                - 'id': The id of the sample
                - 'img_id': The image_id of the sample
                - 'pred_instances': The prediction results of instance(s)
        """
        for data_sample in data_samples:
            if 'pred_instances' not in data_sample:
                raise ValueError(
                    '`pred_instances` are required to process the '
                    f'predictions results in {self.__class__.__name__}. ')

            # keypoints.shape: [N, K, 2],
            # N: number of instances, K: number of keypoints
            # for topdown-style output, N is usually 1, while for
            # bottomup-style output, N is the number of instances in the image
            keypoints = data_sample['pred_instances']['keypoints']
            # [N, K], the scores for all keypoints of all instances
            keypoint_scores = data_sample['pred_instances']['keypoint_scores']
            assert keypoint_scores.shape == keypoints.shape[:2]

            # parse prediction results
            pred = dict()
            pred['id'] = data_sample['id']
            pred['img_id'] = data_sample['img_id']

            pred['keypoints'] = keypoints
            pred['keypoint_scores'] = keypoint_scores
            # Normalize category_id to a plain int.  Some datasets (e.g. MPII)
            # store it as a list; xtcocotools requires a hashable scalar.
            _cat_id = data_sample.get('category_id', 1)
            try:
                _ndim = getattr(_cat_id, 'ndim', -1)
                if _ndim == 0:
                    _cat_id = int(_cat_id)
                elif len(_cat_id) > 0:
                    _cat_id = int(_cat_id[0])
                else:
                    _cat_id = 1
            except TypeError:
                pass
            pred['category_id'] = int(_cat_id)
            if 'bboxes' in data_sample['pred_instances']:
                pred['bbox'] = bbox_xyxy2xywh(
                    data_sample['pred_instances']['bboxes'])

            if 'bbox_scores' in data_sample['pred_instances']:
                # some one-stage models will predict bboxes and scores
                # together with keypoints
                bbox_scores = data_sample['pred_instances']['bbox_scores']
            elif ('bbox_scores' not in data_sample['gt_instances']
                  or len(data_sample['gt_instances']['bbox_scores']) !=
                  len(keypoints)):
                # bottom-up models might output different number of
                # instances from annotation
                bbox_scores = np.ones(len(keypoints))
            else:
                # top-down models use detected bboxes, the scores of which
                # are contained in the gt_instances
                bbox_scores = data_sample['gt_instances']['bbox_scores']
            pred['bbox_scores'] = bbox_scores

            # Per-prediction area (for small/medium/large buckets).  Use GT
            # bbox_scales only when they align 1:1 with predictions (mock-
            # detector mode); otherwise derive from prediction bboxes.
            n_pred = len(keypoints)
            if 'bbox_scales' in data_sample['gt_instances']:
                gt_scales = data_sample['gt_instances']['bbox_scales']
                if len(gt_scales) == n_pred:
                    pred['areas'] = np.prod(gt_scales, axis=1)
            if 'areas' not in pred and 'bbox' in pred:
                pred['areas'] = pred['bbox'][:, 2] * pred['bbox'][:, 3]

            # parse gt
            gt = dict()
            if self.coco is None:
                gt['width'] = data_sample['ori_shape'][1]
                gt['height'] = data_sample['ori_shape'][0]
                gt['img_id'] = data_sample['img_id']
                if self.iou_type == 'keypoints_crowd':
                    assert 'crowd_index' in data_sample, \
                        '`crowd_index` is required when `self.iou_type` is ' \
                        '`keypoints_crowd`'
                    gt['crowd_index'] = data_sample['crowd_index']
                if self.gt_from_samples:
                    # Always synthesize when gt_from_samples=True so that
                    # missing fields (e.g. 'area' absent in CrowdPose
                    # raw_ann_info) are filled in from gt_instances.
                    gt['raw_ann_info'] = self._synthesize_raw_ann_info(
                        data_sample)
                elif 'raw_ann_info' in data_sample:
                    anns = data_sample['raw_ann_info']
                    gt['raw_ann_info'] = (
                        anns if isinstance(anns, list) else [anns])
                else:
                    assert False, \
                        'The row ground truth annotations are required for ' \
                        'evaluation when `ann_file` is not provided'

            # add converted result to the results list
            self.results.append((pred, gt))

    def _synthesize_raw_ann_info(self, data_sample: dict) -> list:
        """Build COCO-format raw_ann_info dicts from gt_instances + metainfo.

        Used when ``gt_from_samples=True`` and the dataset does not set
        ``raw_ann_info`` on each sample (e.g. MPII, which uses its own
        annotation format).  The synthesized dicts contain all fields
        required by :meth:`gt_to_coco_json`.

        Args:
            data_sample (dict): A single data sample produced by the model
                test step.  Must carry ``gt_instances`` (with ``bboxes``,
                optionally ``bbox_scales``, ``keypoints``,
                ``keypoints_visible``) and top-level metainfo keys ``img_id``,
                ``id``, ``category_id``, and ``ori_shape``.

        Returns:
            list[dict]: One COCO annotation dict per GT instance.
        """
        gt_instances = data_sample['gt_instances']
        img_id = int(data_sample['img_id'])

        # ``id`` may be a scalar or a list (benchmark_e2e merges per-image)
        sample_ids = data_sample['id']
        if not isinstance(sample_ids, (list, tuple)):
            sample_ids = [sample_ids]
        sample_ids = [int(x) for x in sample_ids]

        # category_id: may be scalar, 0-d numpy array, or a sequence
        cat_id = data_sample.get('category_id', 1)
        try:
            _ndim = getattr(cat_id, 'ndim', -1)
            if _ndim == 0:
                cat_id = int(cat_id)
            elif len(cat_id) > 0:
                cat_id = int(cat_id[0])
            else:
                cat_id = 1
        except TypeError:
            cat_id = int(cat_id)

        # After Evaluator.to_dict(), gt_instances is a plain dict; use .get().
        if isinstance(gt_instances, dict):
            bboxes = gt_instances.get('bboxes')
            bbox_scales = gt_instances.get('bbox_scales')
            keypoints = gt_instances.get('keypoints')
            kpts_vis = gt_instances.get('keypoints_visible')
            orig_areas = gt_instances.get('orig_areas')
            iscrowd_arr = gt_instances.get('iscrowd')
        else:
            bboxes = getattr(gt_instances, 'bboxes', None)
            bbox_scales = getattr(gt_instances, 'bbox_scales', None)
            keypoints = getattr(gt_instances, 'keypoints', None)
            kpts_vis = getattr(gt_instances, 'keypoints_visible', None)
            orig_areas = getattr(gt_instances, 'orig_areas', None)
            iscrowd_arr = getattr(gt_instances, 'iscrowd', None)

        # Area from raw_ann_info (e.g. COCO segmentation-mask area).
        # Takes priority over all other fallbacks so COCO keeps its exact area.
        _raw_ann = data_sample.get('raw_ann_info')
        if _raw_ann is not None:
            if isinstance(_raw_ann, dict):
                _raw_ann = [_raw_ann]
            raw_ann_areas: list = [
                float(r['area']) for r in _raw_ann if 'area' in r
            ]
            if not raw_ann_areas:
                raw_ann_areas = None
        else:
            raw_ann_areas = None

        # Per-sample area from PackPoseInputs meta_keys (e.g. MPII's
        # item['area'] = 0.53 * bbox_area).  This provides a dataset-native
        # area for pipelines that don't set orig_areas on gt_instances (i.e.
        # tools/test.py), keeping it consistent with tools/benchmark_e2e.py.
        _sample_area = data_sample.get('area')
        if _sample_area is not None:
            _sa_ndim = getattr(_sample_area, 'ndim', -1)
            if _sa_ndim == 0 or not hasattr(_sample_area, '__len__'):
                sample_areas: list = [float(_sample_area)]
            else:
                sample_areas = [float(a) for a in _sample_area]
        else:
            sample_areas = None

        n = 0
        if bboxes is not None:
            n = len(bboxes)
        elif keypoints is not None:
            n = len(keypoints)

        if n == 0:
            return []

        # If fewer IDs were supplied than instances, extend with unique values.
        while len(sample_ids) < n:
            sample_ids.append(sample_ids[-1] + 1)

        if raw_ann_areas is not None:
            while len(raw_ann_areas) < n:
                raw_ann_areas.append(raw_ann_areas[-1])
        if sample_areas is not None:
            while len(sample_areas) < n:
                sample_areas.append(sample_areas[-1])

        anns = []
        for i in range(n):
            ann_id = sample_ids[i]

            # bbox in COCO xywh format
            if bboxes is not None:
                bbox_xywh = bbox_xyxy2xywh(
                    bboxes[i:i + 1])[0].tolist()
            else:
                bbox_xywh = [0., 0., 1., 1.]

            # area priority:
            #   1. orig_areas     – set by benchmark_e2e from item['area']
            #   2. raw_ann_areas  – from raw_ann_info (e.g. COCO mask area)
            #   3. sample_areas   – from PackPoseInputs meta_key 'area'
            #                       (e.g. MPII's 0.53×bbox area via test.py)
            #   4. bbox_scales product – padded area
            #   5. bbox w×h
            if orig_areas is not None:
                area = float(orig_areas[i])
            elif raw_ann_areas is not None:
                area = raw_ann_areas[i]
            elif sample_areas is not None:
                area = sample_areas[i]
            elif bbox_scales is not None:
                area = float(np.prod(bbox_scales[i]))
            else:
                area = float(bbox_xywh[2] * bbox_xywh[3])

            # Flatten keypoints to COCO format [x, y, v, x, y, v, ...]
            kpts_flat = []
            num_visible = 0
            if keypoints is not None:
                kp_xy = keypoints[i]  # (K, 2)
                if kpts_vis is not None:
                    kv = kpts_vis[i]  # (K,) or (K, 2)
                    if kv.ndim > 1:
                        kv = kv[:, 0]  # first channel = visibility
                    kv = np.round(kv).clip(0, 1).astype(np.int32)
                else:
                    kv = np.ones(len(kp_xy), dtype=np.int32)
                num_visible = int(np.sum(kv > 0))
                kpts_flat = np.stack(
                    [kp_xy[:, 0], kp_xy[:, 1],
                     kv.astype(np.float32)],
                    axis=1).flatten().tolist()

            # Use real iscrowd when available (set by unified benchmark loader).
            if iscrowd_arr is not None and i < len(iscrowd_arr):
                iscrowd = int(iscrowd_arr[i])
            else:
                iscrowd = 0

            ann = dict(
                id=ann_id,
                image_id=img_id,
                category_id=cat_id,
                bbox=bbox_xywh,
                keypoints=kpts_flat,
                num_keypoints=num_visible,
                iscrowd=iscrowd,
                area=area,
            )
            anns.append(ann)

        return anns

    def gt_to_coco_json(self, gt_dicts: Sequence[dict],
                        outfile_prefix: str) -> str:
        """Convert ground truth to coco format json file.

        Args:
            gt_dicts (Sequence[dict]): Ground truth of the dataset. Each dict
                contains the ground truth information about the data sample.
                Required keys of the each `gt_dict` in `gt_dicts`:
                    - `img_id`: image id of the data sample
                    - `width`: original image width
                    - `height`: original image height
                    - `raw_ann_info`: the raw annotation information
                Optional keys:
                    - `crowd_index`: measure the crowding level of an image,
                        defined in CrowdPose dataset
                It is worth mentioning that, in order to compute `CocoMetric`,
                there are some required keys in the `raw_ann_info`:
                    - `id`: the id to distinguish different annotations
                    - `image_id`: the image id of this annotation
                    - `category_id`: the category of the instance.
                    - `bbox`: the object bounding box
                    - `keypoints`: the keypoints cooridinates along with their
                        visibilities. Note that it need to be aligned
                        with the official COCO format, e.g., a list with length
                        N * 3, in which N is the number of keypoints. And each
                        triplet represent the [x, y, visible] of the keypoint.
                    - `iscrowd`: indicating whether the annotation is a crowd.
                        It is useful when matching the detection results to
                        the ground truth.
                There are some optional keys as well:
                    - `area`: it is necessary when `self.use_area` is `True`
                    - `num_keypoints`: it is necessary when `self.iou_type`
                        is set as `keypoints_crowd`.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json file will be named
                "somepath/xxx.gt.json".
        Returns:
            str: The filename of the json file.
        """
        image_infos = []
        annotations = []
        img_ids = []
        ann_ids = []

        for gt_dict in gt_dicts:
            # filter duplicate image_info
            if gt_dict['img_id'] not in img_ids:
                image_info = dict(
                    id=gt_dict['img_id'],
                    width=gt_dict['width'],
                    height=gt_dict['height'],
                )
                if self.iou_type == 'keypoints_crowd':
                    image_info['crowdIndex'] = gt_dict['crowd_index']

                image_infos.append(image_info)
                img_ids.append(gt_dict['img_id'])

            # filter duplicate annotations
            for ann in gt_dict['raw_ann_info']:
                if ann is None:
                    # during evaluation on bottom-up datasets, some images
                    # do not have instance annotation
                    continue

                annotation = dict(
                    id=ann['id'],
                    image_id=ann['image_id'],
                    category_id=ann['category_id'],
                    bbox=ann['bbox'],
                    keypoints=ann['keypoints'],
                    iscrowd=ann['iscrowd'],
                )
                if self.use_area:
                    assert 'area' in ann, \
                        '`area` is required when `self.use_area` is `True`'
                    annotation['area'] = ann['area']

                if self.iou_type == 'keypoints_crowd':
                    assert 'num_keypoints' in ann, \
                        '`num_keypoints` is required when `self.iou_type` ' \
                        'is `keypoints_crowd`'
                    annotation['num_keypoints'] = ann['num_keypoints']

                annotations.append(annotation)
                ann_ids.append(ann['id'])

        categories = self.dataset_meta.get('CLASSES')
        if categories is None:
            # Synthesize a minimal COCO-style category entry when the dataset
            # does not supply one (e.g. MPII).
            kp_names = list(
                self.dataset_meta.get('keypoint_id2name', {}).values())
            skeleton = self.dataset_meta.get('skeleton_links', [])
            categories = [{
                'id': 1,
                'name': 'person',
                'supercategory': 'person',
                'keypoints': kp_names,
                'skeleton': [[s + 1, e + 1] for s, e in skeleton],
            }]

        info = dict(
            date_created=str(datetime.datetime.now()),
            description='Coco json file converted by mmpose CocoMetric.')
        coco_json = dict(
            info=info,
            images=image_infos,
            categories=categories,
            licenses=None,
            annotations=annotations,
        )
        converted_json_path = f'{outfile_prefix}.gt.json'
        dump(coco_json, converted_json_path, sort_keys=True, indent=4)
        return converted_json_path

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        # split prediction and gt list
        preds, gts = zip(*results)

        if self.gt_from_samples:
            # When synthesizing GT from samples, the dataset's metainfo may
            # describe a different keypoint set than the model produces (e.g.
            # MPII dataset_meta with 16 keypoints used alongside a COCO model
            # that outputs 17 keypoints after KeypointConverter).  Detect and
            # fix the mismatch here so that NMS scoring and JSON serialization
            # use the correct keypoint count.
            for p in preds:
                kpts = p.get('keypoints')
                if kpts is not None and len(kpts) > 0:
                    actual_n = kpts.shape[1]  # (N_instances, K, 2)
                    meta_n = self.dataset_meta.get('num_keypoints', actual_n)
                    if actual_n != meta_n:
                        logger.info(
                            f'gt_from_samples: overriding num_keypoints '
                            f'{meta_n} → {actual_n} to match predictions.')
                        self.dataset_meta['num_keypoints'] = actual_n
                        current_sigmas = np.array(
                            self.dataset_meta.get('sigmas', []))
                        if len(current_sigmas) != actual_n:
                            if actual_n == len(self._COCO_SIGMAS):
                                new_sigmas = self._COCO_SIGMAS
                            else:
                                new_sigmas = np.full(actual_n, 0.025)
                            logger.info(
                                f'gt_from_samples: replacing sigmas '
                                f'(len={len(current_sigmas)}) with '
                                f'len={actual_n} sigmas.')
                            self.dataset_meta['sigmas'] = new_sigmas
                    break

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        if self.coco is None:
            # use converted gt json file to initialize coco helper
            logger.info('Converting ground truth to coco format...')
            coco_json_path = self.gt_to_coco_json(
                gt_dicts=gts, outfile_prefix=outfile_prefix)
            self.coco = COCO(coco_json_path)
        if self.gt_converter is not None:
            for id_, ann in self.coco.anns.items():
                self.coco.anns[id_] = transform_ann(
                    ann, self.gt_converter['num_keypoints'],
                    self.gt_converter['mapping'])

        kpts = defaultdict(list)

        # group the preds by img_id
        for pred in preds:
            img_id = pred['img_id']

            if self.pred_converter is not None:
                pred = transform_pred(pred,
                                      self.pred_converter['num_keypoints'],
                                      self.pred_converter['mapping'])

            for idx, keypoints in enumerate(pred['keypoints']):

                instance = {
                    'id': pred['id'],
                    'img_id': pred['img_id'],
                    'category_id': pred['category_id'],
                    'keypoints': keypoints,
                    'keypoint_scores': pred['keypoint_scores'][idx],
                    'bbox_score': pred['bbox_scores'][idx],
                }
                if 'bbox' in pred:
                    instance['bbox'] = pred['bbox'][idx]

                if 'areas' in pred:
                    instance['area'] = pred['areas'][idx]
                else:
                    # use keypoint to calculate bbox and get area
                    area = (
                        np.max(keypoints[:, 0]) - np.min(keypoints[:, 0])) * (
                            np.max(keypoints[:, 1]) - np.min(keypoints[:, 1]))
                    instance['area'] = area

                kpts[img_id].append(instance)

        # sort keypoint results according to id and remove duplicate ones
        kpts = self._sort_and_unique_bboxes(kpts, key='id')

        # score the prediction results according to `score_mode`
        # and perform NMS according to `nms_mode`
        valid_kpts = defaultdict(list)
        if self.pred_converter is not None:
            num_keypoints = self.pred_converter['num_keypoints']
        else:
            num_keypoints = self.dataset_meta['num_keypoints']
        for img_id, instances in kpts.items():
            for instance in instances:
                # concatenate the keypoint coordinates and scores
                instance['keypoints'] = np.concatenate([
                    instance['keypoints'], instance['keypoint_scores'][:, None]
                ],
                                                       axis=-1)
                if self.score_mode == 'bbox':
                    instance['score'] = instance['bbox_score']
                elif self.score_mode == 'keypoint':
                    instance['score'] = np.mean(instance['keypoint_scores'])
                else:
                    bbox_score = instance['bbox_score']
                    if self.score_mode == 'bbox_rle':
                        keypoint_scores = instance['keypoint_scores']
                        instance['score'] = float(bbox_score +
                                                  np.mean(keypoint_scores) +
                                                  np.max(keypoint_scores))

                    else:  # self.score_mode == 'bbox_keypoint':
                        mean_kpt_score = 0
                        valid_num = 0
                        for kpt_idx in range(num_keypoints):
                            kpt_score = instance['keypoint_scores'][kpt_idx]
                            if kpt_score > self.keypoint_score_thr:
                                mean_kpt_score += kpt_score
                                valid_num += 1
                        if valid_num != 0:
                            mean_kpt_score /= valid_num
                        instance['score'] = bbox_score * mean_kpt_score
            # perform nms
            if self.nms_mode == 'none':
                valid_kpts[img_id] = instances
            else:
                nms = oks_nms if self.nms_mode == 'oks_nms' else soft_oks_nms
                keep = nms(
                    instances,
                    self.nms_thr,
                    sigmas=self.dataset_meta['sigmas'])
                valid_kpts[img_id] = [instances[_keep] for _keep in keep]

        # convert results to coco style and dump into a json file
        self.results2json(valid_kpts, outfile_prefix=outfile_prefix)

        # only format the results without doing quantitative evaluation
        if self.format_only:
            logger.info('results are saved in '
                        f'{osp.dirname(outfile_prefix)}')
            return {}

        # evaluation results
        eval_results = OrderedDict()
        logger.info(f'Evaluating {self.__class__.__name__}...')
        info_str = self._do_python_keypoint_eval(outfile_prefix)
        name_value = OrderedDict(info_str)
        eval_results.update(name_value)

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results

    def results2json(self, keypoints: Dict[int, list],
                     outfile_prefix: str) -> str:
        """Dump the keypoint detection results to a COCO style json file.

        Args:
            keypoints (Dict[int, list]): Keypoint detection results
                of the dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            str: The json file name of keypoint results.
        """
        # the results with category_id
        cat_results = []

        for _, img_kpts in keypoints.items():
            _keypoints = np.array(
                [img_kpt['keypoints'] for img_kpt in img_kpts])
            num_keypoints = self.dataset_meta['num_keypoints']
            # collect all the person keypoints in current image
            _keypoints = _keypoints.reshape(-1, num_keypoints * 3)

            result = []
            for img_kpt, keypoint in zip(img_kpts, _keypoints):
                res = {
                    'image_id': img_kpt['img_id'],
                    'category_id': img_kpt['category_id'],
                    'keypoints': keypoint.tolist(),
                    'score': float(img_kpt['score']),
                }
                if 'bbox' in img_kpt:
                    res['bbox'] = img_kpt['bbox'].tolist()
                result.append(res)

            cat_results.extend(result)

        res_file = f'{outfile_prefix}.keypoints.json'
        dump(cat_results, res_file, sort_keys=True, indent=4)

    def _do_python_keypoint_eval(self, outfile_prefix: str) -> list:
        """Do keypoint evaluation using COCOAPI.

        Args:
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.keypoints.json",

        Returns:
            list: a list of tuples. Each tuple contains the evaluation stats
            name and corresponding stats value.
        """
        res_file = f'{outfile_prefix}.keypoints.json'
        coco_det = self.coco.loadRes(res_file)
        sigmas = self.dataset_meta['sigmas']
        coco_eval = COCOeval(self.coco, coco_det, self.iou_type, sigmas,
                             self.use_area)
        coco_eval.params.useSegm = None
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        if self.iou_type == 'keypoints_crowd':
            stats_names = [
                'AP', 'AP .5', 'AP .75', 'AR', 'AR .5', 'AR .75', 'AP(E)',
                'AP(M)', 'AP(H)'
            ]
        else:
            stats_names = [
                'AP', 'AP .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5',
                'AR .75', 'AR (M)', 'AR (L)'
            ]

        info_str = list(zip(stats_names, coco_eval.stats))

        return info_str

    def _sort_and_unique_bboxes(self,
                                kpts: Dict[int, list],
                                key: str = 'id') -> Dict[int, list]:
        """Sort keypoint detection results in each image and remove the
        duplicate ones. Usually performed in multi-batch testing.

        Args:
            kpts (Dict[int, list]): keypoint prediction results. The keys are
                '`img_id`' and the values are list that may contain
                keypoints of multiple persons. Each element in the list is a
                dict containing the ``'key'`` field.
                See the argument ``key`` for details.
            key (str): The key name in each person prediction results. The
                corresponding value will be used for sorting the results.
                Default: ``'id'``.

        Returns:
            Dict[int, list]: The sorted keypoint detection results.
        """
        for img_id, persons in kpts.items():
            # deal with bottomup-style output
            if isinstance(kpts[img_id][0][key], Sequence):
                return kpts
            num = len(persons)
            kpts[img_id] = sorted(kpts[img_id], key=lambda x: x[key])
            for i in range(num - 1, 0, -1):
                if kpts[img_id][i][key] == kpts[img_id][i - 1][key]:
                    del kpts[img_id][i]

        return kpts
