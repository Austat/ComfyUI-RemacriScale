# file: ComfyUI/custom_nodes/ComfyUI-RemacriScale/__init__.py

from .upscale_node import RemacriOnnxUpscaleNode
from .upscale_node_lowvram import RemacriOnnxUpscaleNodeLowVram

NODE_CLASS_MAPPINGS = {
    "RemacriOnnxUpscaleNode": RemacriOnnxUpscaleNode,
    "RemacriOnnxUpscaleNodeLowVram": RemacriOnnxUpscaleNodeLowVram
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemacriOnnxUpscaleNode": "Upscale node (ONNX)",
    "RemacriOnnxUpscaleNodeLowVram": "Upscale node LowVRAM (ONNX)",
}
