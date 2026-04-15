#!/usr/bin/env python3
"""
Export Cutie PyTorch model to 6 ONNX submodules for C++ inference.

Usage:
    python export_onnx.py --variant base --weights cutie-base-mega.pth --output ../model/

Output files:
    pixel_encoder.onnx       - image → f16, f8, f4, pix_feat
    key_projection.onnx      - f16 → key, shrinkage, selection
    mask_encoder.onnx        - image + features + sensory + masks → value, sensory, summaries
    pixel_fuser.onnx         - pix_feat + readout + sensory + last_mask → fused
    object_transformer.onnx  - pixel_readout + obj_memory → updated_readout
    mask_decoder.onnx        - f8 + f4 + readout + sensory → new_sensory, logits
"""

import argparse
import os
import sys

# Redirect PyTorch hub/model cache to /tmp to avoid cluttering home directory
os.environ.setdefault('TORCH_HOME', '/tmp/torch_cache')

import torch
import torch.nn as nn
import torch.nn.functional as F

# 从已安装的 cutie 包获取路径（支持 venv 安装）
import importlib.util as _ilu
_cutie_spec = _ilu.find_spec('cutie')
if _cutie_spec is None:
    raise ImportError("cutie 包未找到，请先在当前 Python 环境中安装 cutie（pip install -e .ref_project/Cutie）")
CUTIE_ROOT = os.path.dirname(_cutie_spec.submodule_search_locations[0])

from cutie.model.cutie import CUTIE
from omegaconf import OmegaConf

# ─── Monkey-patch downsample_groups to use static avg_pool2d ──────────────────
# F.interpolate(..., mode='area') internally calls adaptive_avg_pool2d with a
# dynamic output_size that the ONNX legacy exporter cannot handle.
# For integer downscale ratios (1/2, 1/4) avg_pool2d with matching kernel/stride
# is numerically identical to area interpolation.
import cutie.model.group_modules as _gm
import cutie.model.modules as _mod


def _downsample_groups_onnx(g: torch.Tensor,
                             ratio: float = 1 / 2,
                             mode: str = 'area',
                             align_corners=None) -> torch.Tensor:
    batch_size, num_objects = g.shape[:2]
    kernel = int(round(1.0 / ratio))
    g_flat = F.avg_pool2d(g.flatten(start_dim=0, end_dim=1),
                          kernel_size=kernel, stride=kernel)
    return g_flat.view(batch_size, num_objects, *g_flat.shape[1:])


_gm.downsample_groups = _downsample_groups_onnx
_mod.downsample_groups = _downsample_groups_onnx

# ─── Monkey-patch dynamic-size F.interpolate(mode='area') calls ───────────────
# Two more places use F.interpolate(size=dynamic_target, mode='area'):
#   1. CUTIE.pixel_fusion   — last_mask downsampled to sensory spatial size (1/16)
#   2. ObjectSummarizer.forward — masks downsampled to value spatial size (1/16)
# Both always downscale by exactly 16 from the original resolution, so we can
# replace them with avg_pool2d(kernel_size=16, stride=16) which has a static
# kernel and is ONNX-exportable.
#
# Helper: area-pool a [B, N, H, W] tensor down to (target_h, target_w)
def _area_pool_down(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    """Replace F.interpolate(mode='area', size=(target_h, target_w)) for integer ratios."""
    h_in, w_in = tensor.shape[-2], tensor.shape[-1]
    kh = int(h_in) // int(target_h)
    kw = int(w_in) // int(target_w)
    batch_size, num_objects = tensor.shape[0], tensor.shape[1]
    flat = tensor.flatten(start_dim=0, end_dim=1)   # [B*N, H, W]
    flat = flat.unsqueeze(1)                         # [B*N, 1, H, W]
    pooled = F.avg_pool2d(flat, kernel_size=(kh, kw), stride=(kh, kw))
    pooled = pooled.squeeze(1)                       # [B*N, h, w]
    return pooled.view(batch_size, num_objects, *pooled.shape[-2:])


# Patch 1: CUTIE.pixel_fusion
import cutie.model.cutie as _cutie_mod

def _patched_pixel_fusion(self, pix_feat, pixel, sensory, last_mask, *, chunk_size=-1):
    target_h = int(sensory.shape[-2])
    target_w = int(sensory.shape[-1])
    last_mask = _area_pool_down(last_mask, target_h, target_w)
    last_others = self._get_others(last_mask)
    return self.pixel_fuser(pix_feat, pixel, sensory, last_mask, last_others,
                            chunk_size=chunk_size)

_cutie_mod.CUTIE.pixel_fusion = _patched_pixel_fusion


# Patch 2: ObjectSummarizer.forward
import cutie.model.transformer.object_summarizer as _os_mod

def _patched_os_forward(self, masks, value, need_weights=False):
    h, w = int(value.shape[-2]), int(value.shape[-1])
    masks = _area_pool_down(masks, h, w)
    masks = masks.unsqueeze(-1)
    inv_masks = 1 - masks
    repeated_masks = torch.cat([
        masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
        inv_masks.expand(-1, -1, -1, -1, self.num_summaries // 2),
    ], dim=-1)

    value = value.permute(0, 1, 3, 4, 2)
    value = self.input_proj(value)
    if self.add_pe:
        pe = self.pos_enc(value)
        value = value + pe

    with torch.cuda.amp.autocast(enabled=False):
        value = value.float()
        feature = self.feature_pred(value)
        logits = self.weights_pred(value)
        sums, area = _os_mod._weighted_pooling(repeated_masks, feature, logits)

    summaries = torch.cat([sums, area], dim=-1)
    if need_weights:
        return summaries, logits
    return summaries, None

_os_mod.ObjectSummarizer.forward = _patched_os_forward


# ─── Monkey-patch PositionalEncoding for dynamic spatial dims ─────────────────
# The original forward uses torch.arange(h) and torch.zeros((h, w, ...)) which
# get frozen as constants during ONNX tracing (e.g. h=30 when exporting at 480).
# At runtime with a different resolution the hardcoded shapes cause mismatches.
#
# Fix: derive position indices from the input tensor's spatial dimensions via
# ones_like + cumsum (ONNX: Shape→ConstantOfShape→CumSum — fully dynamic), and
# assemble the [H, W, dim*2] embedding via broadcast-add-zero + cat instead of
# indexed assignment into a constant-shaped zeros tensor.
import cutie.model.transformer.positional_encoding as _pe_mod

def _patched_pe_forward(self, tensor: torch.Tensor) -> torch.Tensor:
    if len(tensor.shape) != 4 and len(tensor.shape) != 5:
        raise RuntimeError(f'The input tensor has to be 4/5d, got {tensor.shape}!')

    if len(tensor.shape) == 5:
        num_objects = tensor.shape[1]
        tensor = tensor[:, 0]
    else:
        num_objects = None

    if self.channel_last:
        batch_size, h, w, c = tensor.shape
    else:
        batch_size, c, h, w = tensor.shape

    # ── Dynamic position indices via ones_like + cumsum ──
    # Slice the input to get reference tensors whose ONNX shape is linked to
    # the input's dynamic H/W dimensions.  ones_like produces a
    # Shape→ConstantOfShape node, keeping the length dynamic in the ONNX graph.
    if self.channel_last:
        ref_h = tensor[0, :, 0, 0]   # [H]
        ref_w = tensor[0, 0, :, 0]   # [W]
    else:
        ref_h = tensor[0, 0, :, 0]   # [H]
        ref_w = tensor[0, 0, 0, :]   # [W]

    pos_y = (torch.cumsum(torch.ones_like(ref_h), dim=0) - 1.0).to(self.inv_freq.dtype)
    pos_x = (torch.cumsum(torch.ones_like(ref_w), dim=0) - 1.0).to(self.inv_freq.dtype)

    if self.normalize:
        pos_y = pos_y / (pos_y[-1] + self.eps) * self.scale
        pos_x = pos_x / (pos_x[-1] + self.eps) * self.scale

    sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
    sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
    emb_y = _pe_mod.get_emb(sin_inp_y).unsqueeze(1)   # [H, 1, dim]
    emb_x = _pe_mod.get_emb(sin_inp_x)                 # [W, dim]

    # ── Dynamic [H, W, dim*2] assembly ──
    # Broadcast each component to [H, W, dim] by adding zeros_like of the other
    # (zeros_like → Shape→ConstantOfShape, Add → broadcast: all dynamic).
    emb_x_3d = emb_x.unsqueeze(0)                       # [1, W, dim]
    emb_x_hw = emb_x_3d + torch.zeros_like(emb_y)       # [H, W, dim]
    emb_y_hw = emb_y + torch.zeros_like(emb_x_3d)       # [H, W, dim]
    emb = torch.cat([emb_x_hw, emb_y_hw], dim=-1)       # [H, W, dim*2]

    if not self.channel_last and self.transpose_output:
        pass   # cancelled out
    elif (not self.channel_last) or (self.transpose_output):
        emb = emb.permute(2, 0, 1)

    penc = emb.unsqueeze(0).expand(batch_size, -1, -1, -1)
    if num_objects is None:
        return penc
    else:
        return penc.unsqueeze(1)

_pe_mod.PositionalEncoding.forward = _patched_pe_forward
# ──────────────────────────────────────────────────────────────────────────────


def load_model(variant: str, weights_path: str, device: str = 'cpu') -> CUTIE:
    """Load a Cutie model with pretrained weights."""
    cfg_path = os.path.join(CUTIE_ROOT, 'cutie', 'config', 'model', f'{variant}.yaml')
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    model_cfg = OmegaConf.create({'model': OmegaConf.load(cfg_path)})
    network = CUTIE(model_cfg).to(device).eval()

    if weights_path and os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=False)
        if 'network' in state:
            state = state['network']
        network.load_state_dict(state, strict=True)
        print(f"Loaded weights from {weights_path}")
    else:
        print(f"WARNING: No weights loaded (path: {weights_path})")

    return network


# ═══════════════════════════════════════════════════════════════════
# Wrapper modules for clean ONNX export
# ═══════════════════════════════════════════════════════════════════

class PixelEncoderWrapper(nn.Module):
    def __init__(self, network: CUTIE):
        super().__init__()
        self.pixel_encoder = network.pixel_encoder
        self.pix_feat_proj = network.pix_feat_proj

    def forward(self, image):
        # image: [1, 3, H, W]
        ms_features = self.pixel_encoder(image)
        # ms_features = [f16, f8, f4]
        pix_feat = self.pix_feat_proj(ms_features[0])
        return ms_features[0], ms_features[1], ms_features[2], pix_feat


class KeyProjectionWrapper(nn.Module):
    def __init__(self, network: CUTIE):
        super().__init__()
        self.key_proj = network.key_proj

    def forward(self, f16):
        # f16: [1, C16, H/16, W/16]
        key, shrinkage, selection = self.key_proj(f16, need_s=True, need_e=True)
        return key, shrinkage, selection


class MaskEncoderWrapper(nn.Module):
    def __init__(self, network: CUTIE):
        super().__init__()
        self.mask_encoder = network.mask_encoder
        self.object_summarizer = network.object_summarizer

    def forward(self, image, pix_feat, sensory, masks):
        # image:    [1, 3, H, W]
        # pix_feat: [1, pixel_dim, h16, w16]  (projected 256-ch feature)
        # sensory:  [1, N, sensory_dim, h16, w16]
        # masks:    [1, N, H, W]
        others = (masks.sum(dim=1, keepdim=True) - masks).clamp(0, 1)
        msk_value, new_sensory = self.mask_encoder(
            image, pix_feat, sensory, masks, others,
            deep_update=True, chunk_size=-1)
        obj_summaries, _ = self.object_summarizer(masks, msk_value)
        return msk_value, new_sensory, obj_summaries


class PixelFuserWrapper(nn.Module):
    def __init__(self, network: CUTIE):
        super().__init__()
        self.pixel_fuser = network.pixel_fuser

    def forward(self, pix_feat, pixel, sensory, last_mask):
        # pix_feat:  [1, pixel_dim, h16, w16]
        # pixel:     [1, N, value_dim, h16, w16]
        # sensory:   [1, N, sensory_dim, h16, w16]
        # last_mask: [1, N, h16, w16]  (already at 1/16 scale)
        last_others = (last_mask.sum(dim=1, keepdim=True) - last_mask).clamp(0, 1)
        fused = self.pixel_fuser(pix_feat, pixel, sensory, last_mask, last_others)
        return fused


class ObjectTransformerWrapper(nn.Module):
    def __init__(self, network: CUTIE):
        super().__init__()
        self.object_transformer = network.object_transformer

    def forward(self, pixel_readout, obj_memory):
        # pixel_readout: [1, N, embed_dim, h16, w16]
        # obj_memory:    [1, N, T=1, num_queries, embed_dim+1]
        result, _ = self.object_transformer(pixel_readout, obj_memory)
        return result


class MaskDecoderWrapper(nn.Module):
    def __init__(self, network: CUTIE):
        super().__init__()
        self.mask_decoder = network.mask_decoder

    def forward(self, f8, f4, memory_readout, sensory):
        # f8:             [1, C8, h8, w8]
        # f4:             [1, C4, h4, w4]
        # memory_readout: [1, N, embed_dim, h16, w16]
        # sensory:        [1, N, sensory_dim, h16, w16]
        new_sensory, logits = self.mask_decoder(
            [None, f8, f4], memory_readout, sensory)
        return new_sensory, logits


def export_submodule(module, input_examples, output_path, input_names, output_names,
                     dynamic_axes=None, opset=17):
    """Export a PyTorch module to ONNX."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            module,
            tuple(input_examples),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes or {},
            opset_version=opset,
            do_constant_folding=True,
        )
    print(f"  Exported: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Export Cutie to ONNX submodules')
    parser.add_argument('--variant', choices=['base', 'small'], default='base')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to .pth weights file')
    parser.add_argument('--output', type=str, default='./onnx_models/',
                        help='Output directory for ONNX files')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--num-objects', type=int, default=2)
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()

    device = 'cpu'
    network = load_model(args.variant, args.weights, device)

    # Derive prefix from weights filename: "cutie-base-mega.pth" → "cutie-base-mega_"
    weights_stem = os.path.splitext(os.path.basename(args.weights))[0]
    prefix = weights_stem + '_'

    H, W = args.height, args.width
    h16, w16 = H // 16, W // 16
    h8, w8 = H // 8, W // 8
    h4, w4 = H // 4, W // 4
    n_obj = args.num_objects

    # Dimensions from config
    if args.variant == 'base':
        c16, c8, c4 = 1024, 512, 256
    else:
        c16, c8, c4 = 256, 128, 64
    pixel_dim = 256
    key_dim = 64
    value_dim = 256
    sensory_dim = 256
    embed_dim = 256
    num_queries = 16

    print(f"\nExporting {args.variant} model to {args.output}")
    print(f"  Resolution: {H}×{W} (used as trace example; model supports dynamic resolution), Objects: {n_obj}\n")

    # ── Dynamic-axes specs for each submodule ──────────────────────
    # All spatial dimensions (H/W at each scale) are marked dynamic so
    # the exported ONNX can accept any resolution at runtime.
    dyn_pe = {
        'image':    {2: 'h',   3: 'w'},
        'f16':      {2: 'h16', 3: 'w16'},
        'f8':       {2: 'h8',  3: 'w8'},
        'f4':       {2: 'h4',  3: 'w4'},
        'pix_feat': {2: 'h16', 3: 'w16'},
    }
    dyn_kp = {
        'f16':       {2: 'h16', 3: 'w16'},
        'key':       {2: 'h16', 3: 'w16'},
        'shrinkage': {2: 'h16', 3: 'w16'},
        'selection': {2: 'h16', 3: 'w16'},
    }
    dyn_me = {
        'image':        {2: 'h',   3: 'w'},
        'pix_feat':     {2: 'h16', 3: 'w16'},
        'sensory':      {3: 'h16', 4: 'w16'},
        'masks':        {2: 'h',   3: 'w'},
        'mask_value':   {3: 'h16', 4: 'w16'},
        'new_sensory':  {3: 'h16', 4: 'w16'},
        'obj_summaries': {},  # [1, N, 1, Q, ED+1] — no spatial dims
    }
    dyn_pf = {
        'pix_feat':  {2: 'h16', 3: 'w16'},
        'pixel':     {3: 'h16', 4: 'w16'},
        'sensory':   {3: 'h16', 4: 'w16'},
        'last_mask': {2: 'h16', 3: 'w16'},
        'fused':     {3: 'h16', 4: 'w16'},
    }
    dyn_ot = {
        'pixel_readout':   {3: 'h16', 4: 'w16'},
        'obj_memory':      {},  # [1, N, 1, Q, ED+1] — no spatial dims
        'updated_readout': {3: 'h16', 4: 'w16'},
    }
    dyn_md = {
        'f8':             {2: 'h8',  3: 'w8'},
        'f4':             {2: 'h4',  3: 'w4'},
        'memory_readout': {3: 'h16', 4: 'w16'},
        'sensory':        {3: 'h16', 4: 'w16'},
        'new_sensory':    {3: 'h16', 4: 'w16'},
        'logits':         {2: 'h4',  3: 'w4'},
    }
    # ──────────────────────────────────────────────────────────────

    # 1. Pixel Encoder
    print("[1/6] pixel_encoder")
    pe = PixelEncoderWrapper(network).eval()
    img = torch.randn(1, 3, H, W)
    export_submodule(pe, [img],
                     os.path.join(args.output, f'{prefix}pixel_encoder.onnx'),
                     ['image'], ['f16', 'f8', 'f4', 'pix_feat'],
                     dynamic_axes=dyn_pe,
                     opset=args.opset)

    # 2. Key Projection
    print("[2/6] key_projection")
    kp = KeyProjectionWrapper(network).eval()
    f16 = torch.randn(1, c16, h16, w16)
    export_submodule(kp, [f16],
                     os.path.join(args.output, f'{prefix}key_projection.onnx'),
                     ['f16'], ['key', 'shrinkage', 'selection'],
                     dynamic_axes=dyn_kp,
                     opset=args.opset)

    # 3. Mask Encoder
    print("[3/6] mask_encoder")
    me = MaskEncoderWrapper(network).eval()
    export_submodule(me,
                     [torch.randn(1, 3, H, W),
                      torch.randn(1, pixel_dim, h16, w16),
                      torch.randn(1, n_obj, sensory_dim, h16, w16),
                      torch.randn(1, n_obj, H, W)],
                     os.path.join(args.output, f'{prefix}mask_encoder.onnx'),
                     ['image', 'pix_feat', 'sensory', 'masks'],
                     ['mask_value', 'new_sensory', 'obj_summaries'],
                     dynamic_axes=dyn_me,
                     opset=args.opset)

    # 4. Pixel Fuser
    print("[4/6] pixel_fuser")
    pf = PixelFuserWrapper(network).eval()
    export_submodule(pf,
                     [torch.randn(1, pixel_dim, h16, w16),
                      torch.randn(1, n_obj, value_dim, h16, w16),
                      torch.randn(1, n_obj, sensory_dim, h16, w16),
                      torch.randn(1, n_obj, h16, w16)],
                     os.path.join(args.output, f'{prefix}pixel_fuser.onnx'),
                     ['pix_feat', 'pixel', 'sensory', 'last_mask'],
                     ['fused'],
                     dynamic_axes=dyn_pf,
                     opset=args.opset)

    # 5. Object Transformer
    print("[5/6] object_transformer")
    ot = ObjectTransformerWrapper(network).eval()
    export_submodule(ot,
                     [torch.randn(1, n_obj, embed_dim, h16, w16),
                      torch.randn(1, n_obj, 1, num_queries, embed_dim + 1)],
                     os.path.join(args.output, f'{prefix}object_transformer.onnx'),
                     ['pixel_readout', 'obj_memory'],
                     ['updated_readout'],
                     dynamic_axes=dyn_ot,
                     opset=args.opset)

    # 6. Mask Decoder
    print("[6/6] mask_decoder")
    md = MaskDecoderWrapper(network).eval()
    export_submodule(md,
                     [torch.randn(1, c8, h8, w8),
                      torch.randn(1, c4, h4, w4),
                      torch.randn(1, n_obj, embed_dim, h16, w16),
                      torch.randn(1, n_obj, sensory_dim, h16, w16)],
                     os.path.join(args.output, f'{prefix}mask_decoder.onnx'),
                     ['f8', 'f4', 'memory_readout', 'sensory'],
                     ['new_sensory', 'logits'],
                     dynamic_axes=dyn_md,
                     opset=args.opset)

    print(f"\nDone! All 6 ONNX files saved to {args.output}")


if __name__ == '__main__':
    main()
