# file: ComfyUI/custom_nodes/ComfyUI-RemacriScale/__init__.py
from .remacri_node import RemacriOnnxUpscaleNode

NODE_CLASS_MAPPINGS = {
    "RemacriOnnxUpscale": RemacriOnnxUpscaleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemacriOnnxUpscale": "4x Foolhardy Remacri Upscale (ONNX)",
}
