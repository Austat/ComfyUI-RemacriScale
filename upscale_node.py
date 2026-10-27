from .assets.node import RemacriOnnxUpscaleNode

NODE_CLASS_MAPPINGS = {
    "RemacriOnnxUpscaleNode": RemacriOnnxUpscaleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemacriOnnxUpscaleNode": "Remacri Upscale (Multi-GPU ONNX)",
}

__all__ = [
    "RemacriOnnxUpscaleNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]