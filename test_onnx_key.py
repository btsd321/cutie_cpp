#!/usr/bin/env python3
"""
Test ONNX key_projection output vs PyTorch
"""
import torch
import numpy as np
import onnxruntime as ort
import sys
import os

# Add Cutie to path
sys.path.insert(0, '/home/lixinlong/Project/linden_perception/.ref_project/Cutie')

from cutie.utils.get_default_model import get_default_model

# Load PyTorch model
print("Loading PyTorch model...")
cutie = get_default_model()
cutie.eval()

# Load ONNX model
model_dir = '/home/lixinlong/Project/cutie_cpp/share/model'
print("Loading ONNX model...")
key_proj_onnx = ort.InferenceSession(
    f'{model_dir}/cutie-base-mega_key_projection.onnx',
    providers=['CUDAExecutionProvider']
)

# Create test input
f16 = torch.randn(1, 1024, 30, 40).cuda()

# PyTorch forward
print("\n=== PyTorch ===")
with torch.inference_mode():
    key_pt, shrinkage_pt, selection_pt = cutie.key_proj(f16, need_s=True, need_e=True)

print(f"key shape: {key_pt.shape}")
print(f"key range: [{key_pt.min():.4f}, {key_pt.max():.4f}]")
print(f"key mean: {key_pt.mean():.4f}")
print(f"key std: {key_pt.std():.4f}")

# ONNX forward
print("\n=== ONNX ===")
f16_np = f16.cpu().numpy()
key_onnx, shrinkage_onnx, selection_onnx = key_proj_onnx.run(None, {'f16': f16_np})

print(f"key shape: {key_onnx.shape}")
print(f"key range: [{key_onnx.min():.4f}, {key_onnx.max():.4f}]")
print(f"key mean: {key_onnx.mean():.4f}")
print(f"key std: {key_onnx.std():.4f}")

# Compare
print("\n=== Comparison ===")
key_pt_np = key_pt.cpu().numpy()
diff = np.abs(key_pt_np - key_onnx)
print(f"Max diff: {diff.max():.6f}")
print(f"Mean diff: {diff.mean():.6f}")
print(f"Relative error: {(diff / (np.abs(key_pt_np) + 1e-8)).mean():.6f}")

if diff.max() < 1e-4:
    print("\n✓ ONNX output matches PyTorch!")
else:
    print(f"\n✗ ONNX output differs from PyTorch (max diff: {diff.max():.6f})")
