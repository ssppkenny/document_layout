"""
Export doctr fast_base detection model to ONNX.

Run with:
    pixi run python export_doctr_onnx.py

Output: fast_base.onnx  (~19 MB)
"""

import torch
from doctr.models import fast_base
from doctr.models.utils import export_model_to_onnx

print("Loading fast_base (pretrained=True, exportable=True)...")
model = fast_base(pretrained=True, exportable=True)
model.eval()

dummy = torch.zeros(1, 3, 1024, 1024, dtype=torch.float32)

print("Exporting to ONNX...")
export_model_to_onnx(model, "fast_base", dummy_input=dummy)
print("Saved: fast_base.onnx")

# Quick sanity check with cv2.dnn
import cv2
import numpy as np

print("Verifying with cv2.dnn...")
net = cv2.dnn.readNetFromONNX("fast_base.onnx")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

blob = np.zeros((1, 3, 1024, 1024), dtype=np.float32)
net.setInput(blob)
out = net.forward()
print(f"Output shape: {out.shape}")  # expect (1, 1, 1024, 1024)
assert out.shape == (1, 1, 1024, 1024), f"Unexpected shape: {out.shape}"
print("OK — fast_base.onnx is ready.")
