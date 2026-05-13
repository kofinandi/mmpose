"""Compatibility shims to allow importing external/PETR (Opera toolkit,
mmcv v1.x + mmdet v2.x based) in an mmpose v1.x + mmcv 2.x + mmdet 3.x
environment without modifying the submodule.

Call ``install_petr_shims()`` once before importing any Opera model code.
"""

import copy
import functools
import inspect
import logging
import sys
import types

import torch
import torch.nn as nn
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_dist_info():
    if not dist.is_available() or not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def _noop_fp_decorator(fn=None, **deco_kwargs):
    """No-op replacement for force_fp32 / auto_fp16."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapper
    if fn is not None:
        return decorator(fn)
    return decorator


def _noop_jit(fn=None, **kwargs):
    """No-op replacement for mmcv.jit decorator (parrots-only in v1)."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **fkwargs):
            return func(*args, **fkwargs)
        return wrapper
    if fn is not None:
        return decorator(fn)
    return decorator


# ---------------------------------------------------------------------------
# Stub eval hooks (mmcv.runner in v1)
# ---------------------------------------------------------------------------

class _EvalHook:
    def __init__(self, *a, **kw):
        pass


class _DistEvalHook(_EvalHook):
    pass


# ---------------------------------------------------------------------------
# Transformer base classes for mmdet.models.utils.transformer shim
# (PETRTransformer inherits from Transformer; SOITTransformer from
# DeformableDetrTransformer -- but only PETR is used at inference time)
# ---------------------------------------------------------------------------

def _build_transformer_layer_seq(cfg):
    # Prefer opera's builder which has the opera registry in its parent chain.
    # Fall back to mmcv's if opera hasn't been imported yet.
    try:
        from opera.models.utils.builder import build_transformer_layer_sequence
        return build_transformer_layer_sequence(cfg)
    except ImportError:
        from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
        return build_transformer_layer_sequence(cfg)


class _Transformer(nn.Module):
    """Minimal replacement for mmdet v2 Transformer base class."""

    def __init__(self, encoder=None, decoder=None, init_cfg=None, **kwargs):
        super().__init__()
        if encoder is not None:
            self.encoder = _build_transformer_layer_seq(encoder)
        if decoder is not None:
            self.decoder = _build_transformer_layer_seq(decoder)
        if hasattr(self, 'encoder'):
            self.embed_dims = self.encoder.embed_dims

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class _DeformableDetrTransformer(_Transformer):
    """Stub for mmdet v2 DeformableDetrTransformer (only needed for import
    since SOITTransformer subclasses it; SOIT is never instantiated)."""

    def __init__(self, as_two_stage=False, num_feature_levels=4,
                 two_stage_num_proposals=300, **kwargs):
        super().__init__(**kwargs)
        self.as_two_stage = as_two_stage
        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals


# ---------------------------------------------------------------------------
# color_val_matplotlib stub
# ---------------------------------------------------------------------------

def _color_val_matplotlib(color):
    """Convert a color to (R, G, B) tuple in [0, 1] range."""
    import numpy as np
    if isinstance(color, str):
        from matplotlib.colors import to_rgb
        return to_rgb(color)
    if isinstance(color, (tuple, list)):
        if max(color) > 1.0:
            return tuple(c / 255.0 for c in color[:3])
        return tuple(color[:3])
    return (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# CompatRegistry: mmengine Registry that also accepts the old mmcv v1 style
# @registry.register_module  (without parentheses / without calling it)
# ---------------------------------------------------------------------------

def _make_compat_registry_class():
    """Build the CompatRegistry class once mmengine is importable."""
    from mmengine.registry import Registry as _BaseRegistry

    class CompatRegistry(_BaseRegistry):
        """Subclass of mmengine Registry that tolerates the mmcv v1 decorator
        syntax ``@reg.register_module`` (no parentheses)."""

        def register_module(self, name=None, force=False, module=None):
            """Accept both @reg.register_module and @reg.register_module()."""
            # Old-style: @reg.register_module  (class passed directly as name)
            if isinstance(name, type):
                cls = name
                super().register_module(
                    name=cls.__name__, force=force, module=cls)
                return cls
            return super().register_module(name=name, force=force,
                                           module=module)

    return CompatRegistry


# Built lazily so mmengine doesn't need to be importable at module load time.
_CompatRegistry = None


def CompatRegistry(*args, **kwargs):  # noqa: N802 -- behaves like a class
    """Factory that creates CompatRegistry instances.

    Determines scope from the ACTUAL calling code (not from this shim file),
    mimicking how mmengine.Registry infers scope via sys._getframe(2).
    """
    global _CompatRegistry
    if _CompatRegistry is None:
        _CompatRegistry = _make_compat_registry_class()
    if 'scope' not in kwargs:
        # go 1 frame up to reach the code that called CompatRegistry(...)
        frame = sys._getframe(1)
        module = inspect.getmodule(frame)
        if module is not None:
            scope = module.__name__.split('.')[0]
        else:
            scope = 'mmengine'
        kwargs['scope'] = scope
    return _CompatRegistry(*args, **kwargs)


# ---------------------------------------------------------------------------
# mmcv-scoped registry (for type='mmcv.X' lookups in build_from_cfg)
# ---------------------------------------------------------------------------

def _make_deformable_decoder_shim():
    """Build _DeformableDetrTransformerDecoderShim after mmcv is importable.

    Implements the mmcv v1 DeformableDetrTransformerDecoder.forward interface:
    - Accepts ``return_intermediate``, ``valid_ratios``, and ``reg_branches``
    - Expands 2D reference_points to (bs, num_query, num_levels, 2) for each
      cross-attention layer using valid_ratios
    - Applies reg_branches per layer for iterative coordinate refinement
    - Returns (inter_states, inter_references) when return_intermediate=True
    """
    import torch
    from mmcv.cnn.bricks.transformer import TransformerLayerSequence

    class _DeformableDetrTransformerDecoderShim(TransformerLayerSequence):
        def __init__(self, return_intermediate=False, **kwargs):
            super().__init__(**kwargs)
            self.return_intermediate = return_intermediate

        def forward(self, query, *args, reference_points=None,
                    valid_ratios=None, reg_branches=None, **kwargs):
            """Forward matching mmcv v1 DeformableDetrTransformerDecoder.

            Args:
                query (Tensor): (num_query, bs, embed_dims)
                reference_points (Tensor): shape (bs, num_query, 2) or
                    (bs, num_query, 4) in normalised [0,1] coords.
                valid_ratios (Tensor): (bs, num_levels, 2)
                reg_branches (list[Module] | None): per-layer regression heads.
            Returns:
                tuple: (states, references) where states and references are
                    stacked over layers when return_intermediate=True.
            """
            from mmdet.models.layers import inverse_sigmoid
            output = query
            intermediate = []
            intermediate_reference_points = []

            for lid, layer in enumerate(self.layers):
                if reference_points is not None and valid_ratios is not None:
                    if reference_points.shape[-1] == 4:
                        ref_input = reference_points[:, :, None] * torch.cat(
                            [valid_ratios, valid_ratios], -1)[:, None]
                    else:
                        assert reference_points.shape[-1] == 2
                        ref_input = (reference_points[:, :, None]
                                     * valid_ratios[:, None])
                else:
                    ref_input = reference_points

                output = layer(output, *args, reference_points=ref_input,
                               **kwargs)
                output = output.permute(1, 0, 2)

                if reg_branches is not None and lid < len(reg_branches):
                    tmp = reg_branches[lid](output)
                    new_ref = tmp + inverse_sigmoid(reference_points)
                    new_ref = new_ref.sigmoid()
                    reference_points = new_ref.detach()

                output = output.permute(1, 0, 2)

                if self.return_intermediate:
                    intermediate.append(output)
                    intermediate_reference_points.append(reference_points)

            if self.return_intermediate:
                return (torch.stack(intermediate),
                        torch.stack(intermediate_reference_points))
            return output, reference_points

    return _DeformableDetrTransformerDecoderShim


def _build_mmcv_scope_registry(deformable_decoder_shim):
    """Create and populate a registry with scope='mmcv' that maps old mmcv v1
    transformer class names to their mmcv v2 / mmdet v3 equivalents."""
    from mmcv.cnn.bricks.transformer import (
        MODELS as MMCV_MODELS,
        BaseTransformerLayer,
        TransformerLayerSequence,
        MultiScaleDeformableAttention,
        MultiheadAttention,
        FFN,
    )
    from mmengine.registry import Registry as EngineRegistry

    mmcv_scope = EngineRegistry('model', scope='mmcv', parent=MMCV_MODELS)

    # Map mmcv v1 class names -> mmcv v2 equivalents
    mapping = {
        'DetrTransformerEncoder': TransformerLayerSequence,
        'DeformableDetrTransformerEncoder': TransformerLayerSequence,
        'DetrTransformerDecoder': TransformerLayerSequence,
        'DeformableDetrTransformerDecoder': deformable_decoder_shim,
        'DetrTransformerDecoderLayer': BaseTransformerLayer,
        'DeformableDetrTransformerDecoderLayer': BaseTransformerLayer,
        'DetrTransformerEncoderLayer': BaseTransformerLayer,
        'DeformableDetrTransformerEncoderLayer': BaseTransformerLayer,
        'BaseTransformerLayer': BaseTransformerLayer,
        'TransformerLayerSequence': TransformerLayerSequence,
        'MultiScaleDeformableAttention': MultiScaleDeformableAttention,
        'MultiheadAttention': MultiheadAttention,
        'FFN': FFN,
    }

    # SinePositionalEncoding from mmdet v3
    try:
        from mmdet.models.layers import SinePositionalEncoding
        mapping['SinePositionalEncoding'] = SinePositionalEncoding
    except ImportError:
        pass

    for name, cls in mapping.items():
        if name not in mmcv_scope._module_dict:
            mmcv_scope.register_module(name=name, module=cls, force=True)

    return mmcv_scope


# ---------------------------------------------------------------------------
# Master installer
# ---------------------------------------------------------------------------

_shims_installed = False


def install_petr_shims():
    """Patch sys.modules so that external/PETR (Opera) can be imported under
    mmpose v1.x / mmcv 2.x / mmdet 3.x without any modifications to the
    submodule.  Safe to call multiple times; installation happens only once.
    """
    global _shims_installed
    if _shims_installed:
        return
    _shims_installed = True

    import mmcv
    import mmcv.cnn as mmcv_cnn
    import mmcv.cnn.bricks.transformer as mmcv_bricks_t

    # ---- mmcv.jit (no-op; was parrots-only in v1) --------------------------
    mmcv.jit = _noop_jit

    # ---- mmcv.utils --------------------------------------------------------
    from mmengine.registry import build_from_cfg
    utils_mod = types.ModuleType('mmcv.utils')
    # Forward existing mmcv.utils content where it still exists
    try:
        import mmcv.utils as _existing_utils
        for attr in dir(_existing_utils):
            if not attr.startswith('__'):
                setattr(utils_mod, attr, getattr(_existing_utils, attr))
    except Exception:
        pass
    utils_mod.Registry = CompatRegistry
    utils_mod.build_from_cfg = build_from_cfg
    # TORCH_VERSION / digit_version shims used by opera.datasets.builder
    import torch
    utils_mod.TORCH_VERSION = torch.__version__
    utils_mod.digit_version = _digit_version
    utils_mod.print_log = _print_log
    sys.modules['mmcv.utils'] = utils_mod
    mmcv.utils = utils_mod

    # ---- mmcv.runner -------------------------------------------------------
    from mmengine.model import BaseModule
    runner_mod = types.ModuleType('mmcv.runner')
    runner_mod.force_fp32 = _noop_fp_decorator
    runner_mod.auto_fp16 = _noop_fp_decorator
    runner_mod.get_dist_info = _get_dist_info
    runner_mod.BaseModule = BaseModule
    runner_mod.EvalHook = _EvalHook
    runner_mod.DistEvalHook = _DistEvalHook
    sys.modules['mmcv.runner'] = runner_mod
    mmcv.runner = runner_mod

    # mmcv.runner.base_module submodule (imported explicitly in transformer.py)
    base_module_mod = types.ModuleType('mmcv.runner.base_module')
    base_module_mod.BaseModule = BaseModule
    sys.modules['mmcv.runner.base_module'] = base_module_mod
    runner_mod.base_module = base_module_mod

    # ---- mmcv.parallel (stub; only collate is used) ------------------------
    parallel_mod = types.ModuleType('mmcv.parallel')
    parallel_mod.collate = _collate_stub
    sys.modules['mmcv.parallel'] = parallel_mod
    mmcv.parallel = parallel_mod

    # ---- mmcv.cnn additions ------------------------------------------------
    from mmengine.model import (
        bias_init_with_prob, constant_init, normal_init,
        xavier_init, kaiming_init,
    )
    from mmcv.cnn.bricks.transformer import MODELS as MMCV_MODELS
    mmcv_cnn.MODELS = MMCV_MODELS
    mmcv_cnn.bias_init_with_prob = bias_init_with_prob
    mmcv_cnn.constant_init = constant_init
    mmcv_cnn.normal_init = normal_init
    mmcv_cnn.xavier_init = xavier_init
    mmcv_cnn.kaiming_init = kaiming_init

    # Build the DeformableDetrTransformerDecoder shim (needs TransformerLayerSequence)
    _DeformableDetrTransformerDecoderShim = _make_deformable_decoder_shim()

    # ---- mmcv.cnn.bricks.transformer registries ----------------------------
    # mmcv v1 had ATTENTION, POSITIONAL_ENCODING, TRANSFORMER_LAYER_SEQUENCE
    # as separate registries.  v2 unified them under MODELS.  We create
    # *separate* fake parent registries so that opera's child registries can
    # each register scope='opera' without hitting mmengine's "duplicate scope"
    # assertion (each parent can have at most one child per scope).
    from mmengine.registry import Registry as EngineRegistry
    _MMCV_ATTENTION_PARENT = EngineRegistry(
        'attention', scope='mmcv_attention')
    _MMCV_PE_PARENT = EngineRegistry(
        'positional_encoding', scope='mmcv_positional_encoding')
    _MMCV_TLS_PARENT = EngineRegistry(
        'transformer_layer_sequence', scope='mmcv_tls')
    mmcv_bricks_t.ATTENTION = _MMCV_ATTENTION_PARENT
    mmcv_bricks_t.POSITIONAL_ENCODING = _MMCV_PE_PARENT
    mmcv_bricks_t.TRANSFORMER_LAYER_SEQUENCE = _MMCV_TLS_PARENT

    # Register missing transformer classes under the 'mmcv' scope so that
    # configs with type='mmcv.X' resolve correctly via mmengine registry.
    _build_mmcv_scope_registry(_DeformableDetrTransformerDecoderShim)

    # Create 'mmcv'-scoped child registries for each fake parent so that
    # opera.POSITIONAL_ENCODING etc. can find 'mmcv.X' types via parent chain.
    _mmcv_pe_child = EngineRegistry(
        'mmcv_pe_child', scope='mmcv', parent=_MMCV_PE_PARENT)
    try:
        from mmdet.models.layers import SinePositionalEncoding
        _mmcv_pe_child.register_module(
            name='SinePositionalEncoding', module=SinePositionalEncoding,
            force=True)
    except ImportError:
        pass

    _mmcv_attn_child = EngineRegistry(
        'mmcv_attn_child', scope='mmcv', parent=_MMCV_ATTENTION_PARENT)
    from mmcv.cnn.bricks.transformer import (
        MultiScaleDeformableAttention, MultiheadAttention,
        BaseTransformerLayer, TransformerLayerSequence, FFN,
    )
    for _attn_name, _attn_cls in [
        ('MultiScaleDeformableAttention', MultiScaleDeformableAttention),
        ('MultiheadAttention', MultiheadAttention),
        ('BaseTransformerLayer', BaseTransformerLayer),
        ('TransformerLayerSequence', TransformerLayerSequence),
        ('DetrTransformerEncoder', TransformerLayerSequence),
        ('DeformableDetrTransformerDecoder',
         _DeformableDetrTransformerDecoderShim),
        ('DetrTransformerDecoder', TransformerLayerSequence),
        ('DetrTransformerDecoderLayer', BaseTransformerLayer),
        ('FFN', FFN),
    ]:
        _mmcv_attn_child.register_module(
            name=_attn_name, module=_attn_cls, force=True)

    _mmcv_tls_child = EngineRegistry(
        'mmcv_tls_child', scope='mmcv', parent=_MMCV_TLS_PARENT)
    for _name, _cls in [
        ('TransformerLayerSequence', TransformerLayerSequence),
        ('BaseTransformerLayer', BaseTransformerLayer),
        ('DetrTransformerEncoder', TransformerLayerSequence),
        ('DeformableDetrTransformerDecoder',
         _DeformableDetrTransformerDecoderShim),
        ('DetrTransformerDecoder', TransformerLayerSequence),
    ]:
        _mmcv_tls_child.register_module(name=_name, module=_cls, force=True)
    # Also update the mmcv scope registry for MMCV_MODELS
    _mmcv_scope = None
    from mmcv.cnn.bricks.transformer import MODELS as _MMCV_MODELS_REF
    if 'mmcv' in _MMCV_MODELS_REF.children:
        _mmcv_scope = _MMCV_MODELS_REF.children['mmcv']
    if _mmcv_scope is not None:
        _mmcv_scope.register_module(
            name='DeformableDetrTransformerDecoder',
            module=_DeformableDetrTransformerDecoderShim, force=True)

    # ---- mmdet.core (entire module removed in mmdet v3) --------------------
    from mmdet.models.utils.misc import multi_apply
    from mmdet.utils import reduce_mean
    from mmdet.structures.bbox import (
        bbox_mapping_back, bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh,
    )
    from mmdet.structures.bbox import bbox2result
    from mmdet.models import multiclass_nms

    core_mod = _ensure_module('mmdet.core', {
        'multi_apply': multi_apply,
        'reduce_mean': reduce_mean,
        'bbox_mapping_back': bbox_mapping_back,
        'multiclass_nms': multiclass_nms,
        'bbox2result': bbox2result,
        'bbox_cxcywh_to_xyxy': bbox_cxcywh_to_xyxy,
        'bbox_xyxy_to_cxcywh': bbox_xyxy_to_cxcywh,
    })

    # mmdet.core.bbox
    bbox_mod = _ensure_module('mmdet.core.bbox', {
        'bbox2result': bbox2result,
        'bbox_mapping_back': bbox_mapping_back,
        'bbox_cxcywh_to_xyxy': bbox_cxcywh_to_xyxy,
        'bbox_xyxy_to_cxcywh': bbox_xyxy_to_cxcywh,
    })
    core_mod.bbox = bbox_mod

    # mmdet.core.bbox.assigners (and its submodules, imported by path)
    from mmdet.models.task_modules.assigners import BaseAssigner, AssignResult
    assigners_mod = _ensure_module('mmdet.core.bbox.assigners', {
        'BaseAssigner': BaseAssigner,
        'AssignResult': AssignResult,
    })
    bbox_mod.assigners = assigners_mod
    # Sub-modules imported by full path in opera/core/bbox/assigners/*.py
    _ensure_module('mmdet.core.bbox.assigners.assign_result', {
        'AssignResult': AssignResult,
    })
    _ensure_module('mmdet.core.bbox.assigners.base_assigner', {
        'BaseAssigner': BaseAssigner,
    })

    # mmdet.core.bbox.builder  (opera/core/bbox/builder.py uses these as
    # parent registries for its own BBOX_ASSIGNERS/SAMPLERS/CODERS).
    # Each must be a *separate* registry so that opera can add scope='opera'
    # as a child of each without mmengine's "duplicate scope" assertion.
    # We also create a 'mmdet'-scoped child of each fake parent that holds the
    # classes needed at inference time (PseudoSampler etc.).
    from mmdet.models.task_modules.samplers import PseudoSampler
    from mmdet.models.task_modules.assigners import BaseAssigner, AssignResult

    FAKE_BBOX_ASSIGNERS = EngineRegistry(
        'bbox_assigner_parent', scope='mmdet_bbox_assigner')
    _mmdet_assigners_child = EngineRegistry(
        'mmdet_bbox_assigners', scope='mmdet', parent=FAKE_BBOX_ASSIGNERS)
    _mmdet_assigners_child.register_module(
        name='BaseAssigner', module=BaseAssigner, force=True)

    FAKE_BBOX_SAMPLERS = EngineRegistry(
        'bbox_sampler_parent', scope='mmdet_bbox_sampler')
    _mmdet_samplers_child = EngineRegistry(
        'mmdet_bbox_samplers', scope='mmdet', parent=FAKE_BBOX_SAMPLERS)
    _mmdet_samplers_child.register_module(
        name='PseudoSampler', module=PseudoSampler, force=True)

    FAKE_BBOX_CODERS = EngineRegistry(
        'bbox_coder_parent', scope='mmdet_bbox_coder')

    builder_mod = _ensure_module('mmdet.core.bbox.builder', {
        'BBOX_ASSIGNERS': FAKE_BBOX_ASSIGNERS,
        'BBOX_SAMPLERS': FAKE_BBOX_SAMPLERS,
        'BBOX_CODERS': FAKE_BBOX_CODERS,
    })
    bbox_mod.builder = builder_mod

    # mmdet.core.bbox.match_costs (opera defines KptL1Cost, OksCost here)
    FAKE_MATCH_COST = EngineRegistry(
        'match_cost_parent', scope='mmdet_match_cost')
    # Register mmdet match cost classes under 'mmdet' scope so that
    # type='mmdet.FocalLossCost' etc. resolves correctly.
    _mmdet_match_cost_child = EngineRegistry(
        'mmdet_match_costs', scope='mmdet', parent=FAKE_MATCH_COST)
    try:
        from mmdet.models.task_modules.assigners.match_cost import (
            FocalLossCost, ClassificationCost, BBoxL1Cost, IoUCost,
        )
        for _cls in (FocalLossCost, ClassificationCost, BBoxL1Cost, IoUCost):
            _mmdet_match_cost_child.register_module(
                name=_cls.__name__, module=_cls, force=True)
    except ImportError:
        pass
    match_costs_mod = _ensure_module('mmdet.core.bbox.match_costs', {
        'MATCH_COST': FAKE_MATCH_COST,
    })
    bbox_mod.match_costs = match_costs_mod
    match_costs_builder_mod = _ensure_module(
        'mmdet.core.bbox.match_costs.builder', {
            'MATCH_COST': FAKE_MATCH_COST,
        })
    match_costs_mod.builder = match_costs_builder_mod

    # mmdet.core.visualization
    _ensure_module('mmdet.core.visualization', {
        'color_val_matplotlib': _color_val_matplotlib,
    })

    # mmdet.core.post_processing  (inspose_head uses this path)
    _ensure_module('mmdet.core.post_processing', {
        'multiclass_nms': multiclass_nms,
    })

    # ---- mmdet.models.utils.transformer ------------------------------------
    # (opera/models/utils/transformer.py and dense_heads use this)
    from mmdet.models.layers import inverse_sigmoid
    _ensure_module('mmdet.models.utils.transformer', {
        'Transformer': _Transformer,
        'DeformableDetrTransformer': _DeformableDetrTransformer,
        'inverse_sigmoid': inverse_sigmoid,
    })

    # ---- mmdet.models.utils.builder (TRANSFORMER registry) ----------------
    _FAKE_TRANSFORMER = EngineRegistry('Transformer', scope='mmdet_transformer')
    _ensure_module('mmdet.models.utils.builder', {
        'TRANSFORMER': _FAKE_TRANSFORMER,
    })

    # ---- mmdet.models.dense_heads.AnchorFreeHead patch ---------------------
    # mmdet v3 AnchorFreeHead requires loss_by_feat/get_targets and its
    # __init__ requires num_classes/in_channels.  Opera's PETRHead uses
    # super(AnchorFreeHead, self).__init__(init_cfg) to skip AnchorFreeHead
    # and call BaseDenseHead directly.  We replace AnchorFreeHead with a thin
    # wrapper that inherits from BaseDenseHead (not OrigAnchorFreeHead) so the
    # MRO is: PETRHead -> _ConcreteAFH -> BaseDenseHead -> BaseModule.
    import mmdet.models.dense_heads as _mmdet_dh
    import mmdet.models.dense_heads.anchor_free_head as _mmdet_afh_mod
    _OrigAnchorFreeHead = _mmdet_afh_mod.AnchorFreeHead
    _BaseDenseHead = _OrigAnchorFreeHead.__mro__[1]   # BaseDenseHead

    class _ConcreteAnchorFreeHead(_BaseDenseHead):
        """Non-abstract AnchorFreeHead compatible with opera (mmdet v2 API).

        Provides mmdet v2 API stubs on top of mmdet v3's BaseDenseHead so that
        PETRHead (which inherits from this) can call super(AnchorFreeHead, self)
        and get the correct MRO.
        """
        def loss_by_feat(self, *args, **kwargs):
            raise NotImplementedError

        def get_targets(self, *args, **kwargs):
            raise NotImplementedError

        def simple_test(self, feats, img_metas, rescale=False):
            """mmdet v2 BaseDenseHead.simple_test alias.  Delegates to
            simple_test_bboxes if the subclass defines it (PETRHead does)."""
            if hasattr(self, 'simple_test_bboxes'):
                return self.simple_test_bboxes(feats, img_metas, rescale)
            outs = self.forward(feats)
            results_list = self.get_bboxes(*outs, img_metas, rescale=rescale)
            return results_list

    _mmdet_dh.AnchorFreeHead = _ConcreteAnchorFreeHead
    _mmdet_afh_mod.AnchorFreeHead = _ConcreteAnchorFreeHead

    # ---- opera.datasets stub (text_encoder.py imports Objects365) ----------
    if 'opera.datasets' not in sys.modules:
        opera_datasets_stub = types.ModuleType('opera.datasets')
        opera_datasets_stub.Objects365 = type('Objects365', (), {})
        opera_datasets_stub.CocoPoseDataset = type('CocoPoseDataset', (), {'FLIP_PAIRS': []})
        opera_datasets_stub.CrowdPoseDataset = type('CrowdPoseDataset', (), {'FLIP_PAIRS': []})
        sys.modules['opera.datasets'] = opera_datasets_stub
        # Also stub sub-modules to prevent deep imports
        sys.modules['opera.datasets.builder'] = types.ModuleType('opera.datasets.builder')
        sys.modules['opera.datasets.coco_pose'] = types.ModuleType('opera.datasets.coco_pose')
        sys.modules['opera.datasets.crowd_pose'] = types.ModuleType('opera.datasets.crowd_pose')
        sys.modules['opera.datasets.objects365'] = types.ModuleType('opera.datasets.objects365')
        sys.modules['opera.datasets.pipelines'] = types.ModuleType('opera.datasets.pipelines')
        sys.modules['opera.datasets.utils'] = types.ModuleType('opera.datasets.utils')


# ---------------------------------------------------------------------------
# Post-import finalization  (call after `import opera.models`)
# ---------------------------------------------------------------------------

_shims_finalized = False


def finalize_petr_shims():
    """Copy all opera sub-registry entries into opera.MODELS so that MMCV's
    global registry can find them via the 'opera' scope.  Also patch
    opera.PETR._init_layers to a no-op so that mmdet v3
    DetectionTransformer.__init__ does not call the DETR-specific layer init.

    Must be called AFTER all opera modules have been imported (so that
    decorators like @ATTENTION.register_module() have already run).
    Safe to call multiple times.
    """
    global _shims_finalized
    if _shims_finalized:
        return
    _shims_finalized = True

    from opera.models.builder import MODELS as OPERA_MODELS
    from opera.models.utils.builder import (
        ATTENTION, TRANSFORMER, TRANSFORMER_LAYER_SEQUENCE,
    )
    for reg in (ATTENTION, TRANSFORMER, TRANSFORMER_LAYER_SEQUENCE):
        for name, cls in reg._module_dict.items():
            if name not in OPERA_MODELS._module_dict:
                OPERA_MODELS.register_module(name=name, module=cls, force=True)
        # Also propagate children (e.g., opera sub-registries)
        for scope_name, child_reg in getattr(reg, 'children', {}).items():
            if scope_name == 'opera':
                for name, cls in child_reg._module_dict.items():
                    if name not in OPERA_MODELS._module_dict:
                        OPERA_MODELS.register_module(
                            name=name, module=cls, force=True)

    # Patch opera.PETR._init_layers to no-op.
    # opera.PETR.__init__ does super(DETR, self).__init__() which skips
    # DETR.__init__ and calls DetectionTransformer.__init__.  That base class
    # calls self._init_layers(), which without a PETR override falls through to
    # mmdet v3 DETR._init_layers expecting positional_encoding/encoder/decoder
    # at the detector level (which PETR does not have -- they live in PETRHead).
    try:
        from opera.models.detectors.petr import PETR as _PETR
        if not hasattr(_PETR, '_petr_shim_init_layers_patched'):
            _PETR._init_layers = lambda self: None
            _PETR._petr_shim_init_layers_patched = True
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _collate_stub(batch, samples_per_gpu=1):
    """Stub for mmcv.parallel.collate (only needed if opera datasets are used,
    which they are not in the wrapper inference path)."""
    return torch.utils.data.dataloader.default_collate(batch)


def _digit_version(version_str):
    """Parse version string into a comparable tuple."""
    import re
    numbers = re.findall(r'\d+', version_str.split('+')[0])
    return tuple(int(n) for n in numbers[:3])


def _print_log(msg, logger=None, level=logging.INFO):
    logging.getLogger(logger or 'mmcv').log(level, msg)


def _ensure_module(name, attrs=None):
    """Create or retrieve a fake module in sys.modules, optionally setting
    attributes from *attrs*.  Also attaches it to the parent package."""
    if name not in sys.modules:
        mod = types.ModuleType(name)
        sys.modules[name] = mod
        parent_name, _, child_name = name.rpartition('.')
        if parent_name and parent_name in sys.modules:
            setattr(sys.modules[parent_name], child_name, mod)
    if attrs:
        for k, v in attrs.items():
            setattr(sys.modules[name], k, v)
    return sys.modules[name]
