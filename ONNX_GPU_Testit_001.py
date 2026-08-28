# If you have problems runnin the node, you can use this standalon helper- script to debug your problem.
# It will check that your base system is in order. Run this script and *.onnx-file from the same folder.
# The folder can be any folder in your computer.
# You can usually disrecard any warnings, but errors will also result in errors on ComfyUI- node.

# 28.8.2026 -Austat

import onnxruntime as ort
import torch
import os
os.makedirs("./trt_timing_cache", exist_ok=True)
os.makedirs("./trt_engine_cache", exist_ok=True)

print(torch.cuda.mem_get_info())
print(ort.__version__)

# Explicitly list providers from highest optimization to fallback
providers = [
    ("TensorrtExecutionProvider", {
        "trt_fp16_enable": True,               # Enable FP16 precision
        "trt_engine_cache_enable": True,       # Cache the compiled engine to save startup time
        "trt_engine_cache_path": "./trt_cache" # Directory for engine files
    }),
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
]

# Initialize the session
session = ort.InferenceSession("4x_foolhardy_Remacri.onnx", providers=providers)

so = ort.SessionOptions()
session = ort.InferenceSession("4x_foolhardy_Remacri.onnx", sess_options=so, providers=providers)
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Full TensorRT options
trt_options = {
    "trt_fp16_enable": True,
    "trt_engine_cache_enable": True,
    "trt_engine_cache_path": "./trt_engine_cache",
    "trt_timing_cache_enable": True,
    "trt_timing_cache_path": "./trt_timing_cache/test.bin",
}

providers = [
    ("TensorrtExecutionProvider", trt_options),
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
]

print("Creating session with full TRT options…")
session = ort.InferenceSession(
    "4x_foolhardy_Remacri.onnx",
    sess_options=so,
    providers=providers
)

print("OK: Full TRT options")


try:
    session = ort.InferenceSession("4x_foolhardy_Remacri.onnx", providers=[("TensorrtExecutionProvider", trt_options)])
except:
    session = ort.InferenceSession("4x_foolhardy_Remacri.onnx", providers=["CUDAExecutionProvider"])

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    "4x_foolhardy_Remacri.onnx",
    sess_options=so,
    providers=providers
)

print("OK: GraphOptimizationLevel")
