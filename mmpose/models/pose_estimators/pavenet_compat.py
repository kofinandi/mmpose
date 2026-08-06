"""Compatibility shims to allow importing external/PAVENet (Opera toolkit,
mmcv v1.x + mmdet v2.x based -- forked from and structurally identical to
PETR/Opera, see :mod:`petr_compat`) in an mmpose v1.x + mmcv 2.x + mmdet 3.x
environment without modifying the submodule.

PAVE-Net (``opera.PAVE``, ``opera.PAVEHeadMulFrames``) reuses almost all of
PETR's legacy API surface verbatim (same base ``DETR``/``AnchorFreeHead``
MRO tricks, same ``mmdet.core``/``mmcv.runner`` etc. call sites), so this
module builds directly on :func:`petr_compat.install_petr_shims`. It adds
only the pieces that are genuinely new for PAVE-Net's multi-frame ("mul
frames", ``num_frames=3``) architecture and are **not** present anywhere in
the installed mmcv v2 / mmdet v3, but only in PAVENet's private
``third_party/mmcv`` and ``third_party/mmdetection`` forks (which this repo
does not install -- see :mod:`petr_compat`'s module docstring for why):

- A multi-frame-aware ``ResNet`` backbone (``third_party/mmdetection``):
  forward() accepts ``(B, T, C, H, W)`` directly (flattens ``T`` into the
  batch dim before the usual conv stem) when built with
  ``input_type='mul_frames'``. Ported here as a thin subclass of the real
  (installed) ``mmdet.models.backbones.resnet.ResNet`` that behaves
  identically to it when ``input_type`` is not set, so the override is
  safe for any other model (e.g. PETR) sharing the process.
- ``MulFramesMultiScaleDeformableAttentionNumFrames3``
  (``third_party/mmcv/mmcv/ops/multi_scale_deform_attn.py``): a
  deformable-attention variant with three independent sampling-offset /
  attention-weight heads (one per frame in the 3-frame window), fused by
  softmax-normalised per-frame attention mass. Ported using the installed
  mmcv's pure-PyTorch reference sampling kernel
  (``multi_scale_deformable_attn_pytorch``) -- the upstream class only
  uses a custom CUDA kernel as an optional fast path when available and
  otherwise falls back to that exact same reference kernel, so this is
  numerically identical to the released weights' training path, not an
  approximation.
- ``DeformableDetrTransformerDecoderV1``
  (``third_party/mmdetection/mmdet/models/utils/transformer.py``): the
  refinement decoder used by ``with_kpt_refine=True``, which fans a single
  regression branch out into ``pre_reg_branches`` / ``reg_branches`` /
  ``next_reg_branches`` (one per frame). Ported verbatim (pure PyTorch,
  no forked-kernel dependency).
- ``multi_scale_deformable_attn_pytorchV1``: only referenced by a dead
  (never-instantiated, for the released R50/T3 config) alternate attention
  variant in ``opera/models/utils/transformer.py``; stubbed as an alias of
  ``multi_scale_deformable_attn_pytorch`` purely so that *module-level*
  import of the file succeeds.

``finalize_pavenet_shims()`` additionally monkey-patches
``opera.MulFramesMultiScaleDeformablePoseAttentionNumFrames3.vis_attention``
(used by the R50/T3 config's *non*-refine decoder attention, i.e. genuinely
part of the released architecture, unlike the pieces above) to a no-op: the
released code unconditionally calls it from ``forward()`` to render a debug
attention-heatmap overlay by ``cv2.imread()``-ing a hardcoded path from the
authors' own machine, which raises ``AttributeError` as soon as that path
doesn't exist (i.e. unconditionally, in any other environment). It is
read-only w.r.t. the attention tensors (visualisation side effect only), so
disabling it does not change any prediction.

Call ``install_pavenet_shims()`` once before importing any Opera/PAVE model
code, then ``finalize_pavenet_shims()`` after ``from opera.models import
build_model`` has run.
"""

import torch
import torch.nn as nn

from mmpose.models.pose_estimators.petr_compat import (finalize_petr_shims,
                                                        install_petr_shims)

# ---------------------------------------------------------------------------
# Multi-frame ResNet backbone (third_party/mmdetection's ResNet.forward)
# ---------------------------------------------------------------------------


def _make_multi_frame_resnet():
    """Subclass the installed mmdet ResNet, adding PAVENet's
    ``input_type='mul_frames'`` 5D-input support. Behaves exactly like the
    stock class when ``input_type`` is left at its default (``None``), so
    this override is safe to install globally."""
    from mmdet.models.backbones.resnet import ResNet as _MMDetResNet

    class MultiFrameResNet(_MMDetResNet):

        def __init__(self, *args, input_type=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.input_type = input_type

        def forward(self, x):
            if self.input_type == 'mul_frames':
                # (B, T, C, H, W) -> (B*T, C, H, W); PAVEHeadMulFrames
                # keeps this flattened layout throughout (it recovers the
                # per-frame axis itself via strided indexing, e.g.
                # `value[:, :, 0]`/`img_id // self.num_frames`).
                bs, num_frames, c, h, w = x.shape
                x = x.flatten(0, 1)
            return super().forward(x)

    return MultiFrameResNet


# ---------------------------------------------------------------------------
# MulFramesMultiScaleDeformableAttentionNumFrames3 (forked mmcv ops)
# ---------------------------------------------------------------------------


def _make_mul_frames_ms_deform_attn_num_frames3():
    """Port of ``third_party/mmcv/mmcv/ops/multi_scale_deform_attn.py``'s
    ``MulFramesMultiScaleDeformableAttentionNumFrames3``, using the
    installed mmcv's reference (pure PyTorch) sampling kernel unconditionally
    (see module docstring)."""
    from mmcv.cnn import constant_init, xavier_init
    from mmcv.ops.multi_scale_deform_attn import (
        multi_scale_deformable_attn_pytorch)
    from mmcv.runner.base_module import BaseModule

    class MulFramesMultiScaleDeformableAttentionNumFrames3(BaseModule):

        def __init__(self,
                    num_frames=3,
                    embed_dims=256,
                    num_heads=8,
                    num_levels=4,
                    num_points=4,
                    im2col_step=64,
                    dropout=0.1,
                    batch_first=False,
                    norm_cfg=None,
                    init_cfg=None):
            super().__init__(init_cfg)
            if embed_dims % num_heads != 0:
                raise ValueError(
                    f'embed_dims must be divisible by num_heads, but got '
                    f'{embed_dims} and {num_heads}')
            self.norm_cfg = norm_cfg
            self.dropout = nn.Dropout(dropout)
            self.batch_first = batch_first
            self.im2col_step = im2col_step
            self.embed_dims = embed_dims
            self.num_levels = num_levels
            self.num_heads = num_heads
            self.num_points = num_points
            self.num_frames = num_frames
            assert num_frames == 3, (
                'This shim only ports the num_frames=3 variant used by the '
                'released PAVE-Net R50 config.')

            self.pre_sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_levels * num_points * 2)
            self.pre_attention_weights = nn.Linear(
                embed_dims, num_heads * num_levels * num_points)
            self.sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_levels * num_points * 2)
            self.attention_weights = nn.Linear(
                embed_dims, num_heads * num_levels * num_points)
            self.next_sampling_offsets = nn.Linear(
                embed_dims, num_heads * num_levels * num_points * 2)
            self.next_attention_weights = nn.Linear(
                embed_dims, num_heads * num_levels * num_points)

            self.value_proj = nn.Linear(embed_dims, embed_dims)
            self.output_proj = nn.Linear(embed_dims, embed_dims)
            self.init_weights()

        def init_weights(self):
            import math
            constant_init(self.pre_sampling_offsets, 0.)
            constant_init(self.sampling_offsets, 0.)
            constant_init(self.next_sampling_offsets, 0.)
            thetas = torch.arange(
                self.num_heads,
                dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
            grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
            grid_init = (grid_init / grid_init.abs().max(
                -1, keepdim=True)[0]).view(self.num_heads, 1, 1, 2).repeat(
                    1, self.num_levels, self.num_points, 1)
            for i in range(self.num_points):
                grid_init[:, :, i, :] *= i + 1
            self.pre_sampling_offsets.bias.data = grid_init.view(-1).clone()
            self.sampling_offsets.bias.data = grid_init.view(-1).clone()
            self.next_sampling_offsets.bias.data = grid_init.view(-1).clone()
            constant_init(self.pre_attention_weights, val=0., bias=0.)
            constant_init(self.attention_weights, val=0., bias=0.)
            constant_init(self.next_attention_weights, val=0., bias=0.)
            xavier_init(self.value_proj, distribution='uniform', bias=0.)
            xavier_init(self.output_proj, distribution='uniform', bias=0.)
            self._is_init = True

        def forward(self,
                    query,
                    key=None,
                    value=None,
                    identity=None,
                    query_pos=None,
                    key_padding_mask=None,
                    reference_points=None,
                    spatial_shapes=None,
                    level_start_index=None,
                    **kwargs):
            if value is None:
                value = query
            if identity is None:
                identity = query
            if query_pos is not None:
                query = query + query_pos
            if not self.batch_first:
                query = query.permute(1, 0, 2)
                value = value.permute(1, 0, 2, 3)

            bs, num_query, _ = query.shape
            bs, num_value, num_frames, _ = value.shape
            assert (spatial_shapes[:, 0] *
                   spatial_shapes[:, 1]).sum() == num_value

            if key_padding_mask is not None:
                value = value.masked_fill(
                    key_padding_mask.transpose(1, 2)[..., None], 0.0)

            value = self.value_proj(value)
            pre_value = value[:, :, 0].view(bs, num_value, self.num_heads,
                                            -1).contiguous()
            now_value = value[:, :, 1].view(bs, num_value, self.num_heads,
                                            -1).contiguous()
            next_value = value[:, :, 2].view(bs, num_value, self.num_heads,
                                             -1).contiguous()

            pre_sampling_offsets = self.pre_sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels,
                self.num_points, 2)
            now_sampling_offsets = self.sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels,
                self.num_points, 2)
            next_sampling_offsets = self.next_sampling_offsets(query).view(
                bs, num_query, self.num_heads, self.num_levels,
                self.num_points, 2)

            pre_attention_weights = self.pre_attention_weights(query).view(
                bs, num_query, self.num_heads,
                self.num_levels * self.num_points)
            now_attention_weights = self.attention_weights(query).view(
                bs, num_query, self.num_heads,
                self.num_levels * self.num_points)
            next_attention_weights = self.next_attention_weights(
                query).view(bs, num_query, self.num_heads,
                           self.num_levels * self.num_points)

            pre_weights_sum = torch.exp(pre_attention_weights).sum(
                -1, keepdim=True)
            now_weights_sum = torch.exp(now_attention_weights).sum(
                -1, keepdim=True)
            next_weights_sum = torch.exp(next_attention_weights).sum(
                -1, keepdim=True)
            sum_all = pre_weights_sum + now_weights_sum + next_weights_sum

            pre_attention_weights = pre_attention_weights.softmax(
                -1).view(bs, num_query, self.num_heads, self.num_levels,
                        self.num_points)
            now_attention_weights = now_attention_weights.softmax(
                -1).view(bs, num_query, self.num_heads, self.num_levels,
                        self.num_points)
            next_attention_weights = next_attention_weights.softmax(
                -1).view(bs, num_query, self.num_heads, self.num_levels,
                        self.num_points)

            if reference_points.shape[-1] == 2:
                pre_reference_points = reference_points[:bs]
                now_reference_points = reference_points[bs:bs * 2]
                next_reference_points = reference_points[bs * 2:]
                offset_normalizer = torch.stack(
                    [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
                pre_sampling_locations = (
                    pre_reference_points[:, :, None, :, None, :] +
                    pre_sampling_offsets /
                    offset_normalizer[None, None, None, :, None, :])
                now_sampling_locations = (
                    now_reference_points[:, :, None, :, None, :] +
                    now_sampling_offsets /
                    offset_normalizer[None, None, None, :, None, :])
                next_sampling_locations = (
                    next_reference_points[:, :, None, :, None, :] +
                    next_sampling_offsets /
                    offset_normalizer[None, None, None, :, None, :])
            elif reference_points.shape[-1] == 4:
                pre_sampling_locations = (
                    reference_points[:, :, None, :, None, :2] +
                    pre_sampling_offsets / self.num_points *
                    reference_points[:, :, None, :, None, 2:] * 0.5)
                now_sampling_locations = (
                    reference_points[:, :, None, :, None, :2] +
                    now_sampling_offsets / self.num_points *
                    reference_points[:, :, None, :, None, 2:] * 0.5)
                next_sampling_locations = (
                    reference_points[:, :, None, :, None, :2] +
                    next_sampling_offsets / self.num_points *
                    reference_points[:, :, None, :, None, 2:] * 0.5)
            else:
                raise ValueError(
                    'Last dim of reference_points must be 2 or 4, but got '
                    f'{reference_points.shape[-1]} instead.')

            pre_output = multi_scale_deformable_attn_pytorch(
                pre_value, spatial_shapes, pre_sampling_locations,
                pre_attention_weights)
            now_output = multi_scale_deformable_attn_pytorch(
                now_value, spatial_shapes, now_sampling_locations,
                now_attention_weights)
            next_output = multi_scale_deformable_attn_pytorch(
                next_value, spatial_shapes, next_sampling_locations,
                next_attention_weights)

            pre_output = pre_output.reshape(bs, num_query, self.num_heads,
                                            -1)
            now_output = now_output.reshape(bs, num_query, self.num_heads,
                                            -1)
            next_output = next_output.reshape(bs, num_query, self.num_heads,
                                              -1)
            output = (pre_output * (pre_weights_sum / sum_all) +
                     now_output * (now_weights_sum / sum_all) +
                     next_output * (next_weights_sum / sum_all))
            output = output.flatten(-2, -1)
            output = self.output_proj(output)

            if not self.batch_first:
                output = output.permute(1, 0, 2)
            return self.dropout(output) + identity

    return MulFramesMultiScaleDeformableAttentionNumFrames3


# ---------------------------------------------------------------------------
# DeformableDetrTransformerDecoderV1 (forked mmdet transformer utils)
# ---------------------------------------------------------------------------


def _make_deformable_detr_decoder_v1():
    """Port of ``third_party/mmdetection/mmdet/models/utils/transformer.py``'s
    ``DeformableDetrTransformerDecoderV1`` -- the pose-refinement decoder
    used when ``with_kpt_refine=True``, fanning each layer's regression
    output into three per-frame branches (``pre``/``now``/``next``)."""
    from mmcv.cnn.bricks.transformer import TransformerLayerSequence
    from mmdet.models.layers import inverse_sigmoid

    class DeformableDetrTransformerDecoderV1(TransformerLayerSequence):

        def __init__(self, *args, return_intermediate=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.return_intermediate = return_intermediate

        def forward(self,
                   query,
                   *args,
                   reference_points=None,
                   valid_ratios=None,
                   pre_reg_branches=None,
                   reg_branches=None,
                   next_reg_branches=None,
                   **kwargs):
            output = query
            intermediate = []
            intermediate_reference_points = []
            for lid, layer in enumerate(self.layers):
                if reference_points.shape[-1] == 4:
                    reference_points_input = (
                        reference_points[:, :, None] *
                        torch.cat([valid_ratios, valid_ratios],
                                 -1)[:, None])
                else:
                    assert reference_points.shape[-1] == 2
                    reference_points_input = (
                        reference_points[:, :, None] * valid_ratios[:, None])
                output = layer(
                    output,
                    *args,
                    reference_points=reference_points_input,
                    **kwargs)
                output = output.permute(1, 0, 2)

                if reg_branches is not None:
                    pre_tmp = pre_reg_branches[lid](output)
                    tmp = reg_branches[lid](output)
                    next_tmp = next_reg_branches[lid](output)
                    tmps = torch.concat([pre_tmp, tmp, next_tmp], dim=0)
                    if reference_points.shape[-1] == 4:
                        new_reference_points = tmp + inverse_sigmoid(
                            reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    else:
                        assert reference_points.shape[-1] == 2
                        new_reference_points = tmps
                        new_reference_points[
                            ..., :2] = tmps[..., :2] + inverse_sigmoid(
                                reference_points)
                        new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points

                output = output.permute(1, 0, 2)
                if self.return_intermediate:
                    intermediate.append(output)
                    intermediate_reference_points.append(reference_points)

            if self.return_intermediate:
                return torch.stack(intermediate), torch.stack(
                    intermediate_reference_points)
            return output, reference_points

    return DeformableDetrTransformerDecoderV1


# ---------------------------------------------------------------------------
# Master installer
# ---------------------------------------------------------------------------

_pavenet_shims_installed = False


def install_pavenet_shims():
    """Install PETR's shims plus the additional pieces PAVE-Net needs. Safe
    to call multiple times; installation happens only once."""
    global _pavenet_shims_installed
    install_petr_shims()
    if _pavenet_shims_installed:
        return
    _pavenet_shims_installed = True

    import mmcv.cnn.bricks.transformer as mmcv_bricks_t
    import mmcv.ops.multi_scale_deform_attn as ms_deform_attn_mod
    # NOTE: there are *two* distinct 'mmcv'-scope registry hierarchies alive
    # after `install_petr_shims()`, and which one a `type='mmcv.X'` config
    # resolves through depends on *who* calls `build_from_cfg`:
    #  (a) opera's own `build_transformer_layer_sequence`/`build_attention`
    #      (`opera/models/utils/builder.py`) use *opera's own* Registry
    #      objects, whose *parent* is the fake `ATTENTION`/
    #      `TRANSFORMER_LAYER_SEQUENCE` mmcv_bricks_t module attributes that
    #      `install_petr_shims()` overwrote (so that opera's registries can
    #      add an 'opera'-scope child without hitting mmengine's "duplicate
    #      scope" check on the *real* ones). This is the path taken for the
    #      *top-level* `refine_decoder=dict(type='mmcv.
    #      DeformableDetrTransformerDecoderV1', ...)` config (built directly
    #      by `TransformerMulFrames.__init__`).
    #  (b) the real (installed) mmcv v2's own `build_attention`/
    #      `build_transformer_layer` (called from *inside* real
    #      `BaseTransformerLayer`/`TransformerLayerSequence.__init__`, e.g.
    #      for nested `attn_cfgs`) call mmengine's real root `MODELS.build`
    #      directly, which is a *completely different* object from (a) and
    #      is the one `install_petr_shims()` populates via
    #      `_build_mmcv_scope_registry`. This is the path taken for the
    #      *nested* `attn_cfgs=dict(type='mmcv.
    #      MulFramesMultiScaleDeformableAttentionNumFrames3', ...)` above.
    from mmcv.cnn.bricks.transformer import MODELS as _REAL_MMCV_MODELS
    _real_mmcv_scope = _REAL_MMCV_MODELS.children['mmcv']
    _fake_tls_mmcv_scope = mmcv_bricks_t.TRANSFORMER_LAYER_SEQUENCE.children[
        'mmcv']

    # ---- multi_scale_deformable_attn_pytorchV1 (dead-code import target) --
    if not hasattr(ms_deform_attn_mod, 'multi_scale_deformable_attn_pytorchV1'):
        ms_deform_attn_mod.multi_scale_deformable_attn_pytorchV1 = (
            ms_deform_attn_mod.multi_scale_deformable_attn_pytorch)

    # ---- MulFramesMultiScaleDeformableAttentionNumFrames3 (real mmcv scope,
    # nested attn_cfgs lookup -- path (b) above) -----------------------------
    _MulFramesAttn = _make_mul_frames_ms_deform_attn_num_frames3()
    _real_mmcv_scope.register_module(
        name='MulFramesMultiScaleDeformableAttentionNumFrames3',
        module=_MulFramesAttn,
        force=True)

    # ---- DeformableDetrTransformerDecoderV1 (fake mmcv scope, top-level
    # refine_decoder lookup via opera's own builder -- path (a) above) ------
    _DecoderV1 = _make_deformable_detr_decoder_v1()
    _fake_tls_mmcv_scope.register_module(
        name='DeformableDetrTransformerDecoderV1',
        module=_DecoderV1,
        force=True)

    # ---- Multi-frame ResNet backbone (mmdet scope) --------------------------
    from mmdet.registry import MODELS as MMDET_MODELS
    _MultiFrameResNet = _make_multi_frame_resnet()
    MMDET_MODELS.register_module(
        name='ResNet', module=_MultiFrameResNet, force=True)


_pavenet_shims_finalized = False


def finalize_pavenet_shims():
    """Copy PAVE-Net's opera sub-registry entries into opera.MODELS (via
    ``finalize_petr_shims``) and patch ``opera.PAVE._init_layers`` to a
    no-op, mirroring ``finalize_petr_shims``'s treatment of ``opera.PETR``
    (same DETR-skipping MRO issue -- see that function's docstring). Must be
    called AFTER all opera modules have been imported. Safe to call multiple
    times."""
    global _pavenet_shims_finalized
    finalize_petr_shims()
    if _pavenet_shims_finalized:
        return
    _pavenet_shims_finalized = True

    try:
        from opera.models.detectors.pave import PAVE as _PAVE
        if not hasattr(_PAVE, '_pavenet_shim_init_layers_patched'):
            _PAVE._init_layers = lambda self: None
            _PAVE._pavenet_shim_init_layers_patched = True
    except Exception:
        pass

    # ``MulFramesMultiScaleDeformablePoseAttentionNumFrames3.forward``
    # (opera.models.utils.transformer) unconditionally calls
    # ``self.vis_attention(...)`` -- leftover, never-disabled debugging
    # code from the authors' own experiments that ``cv2.imread()``s a
    # hardcoded path on their training machine
    # (``/datasets/17/rename/images_renamed/...``) and crashes with
    # ``AttributeError: 'NoneType' object has no attribute 'shape'`` as
    # soon as that path doesn't exist (i.e. always, outside the authors'
    # own filesystem). It is read-only w.r.t. the attention weights/
    # locations it's given (only used to render a heatmap overlay PNG),
    # so no-opping it does not change any prediction -- purely restores
    # runnability.
    try:
        from opera.models.utils.transformer import (
            MulFramesMultiScaleDeformablePoseAttentionNumFrames3 as _Attn)
        if not hasattr(_Attn, '_pavenet_shim_vis_attention_patched'):
            _Attn.vis_attention = lambda self, *args, **kwargs: None
            _Attn._pavenet_shim_vis_attention_patched = True
    except Exception:
        pass
