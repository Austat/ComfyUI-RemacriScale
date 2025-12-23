# file: ComfyUI/custom_nodes/ComfyUI-RemacriScale/__init__.py
from .upscale_node import RemacriOnnxUpscaleNode

NODE_CLASS_MAPPINGS = {
    "RemacriOnnxUpscaleNode": RemacriOnnxUpscaleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemacriOnnxUpscaleNode": "Upscale node (ONNX)",
}
