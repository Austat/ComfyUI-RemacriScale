from .assets.node import RemacriOnnxUpscaleNode

NODE_CLASS_MAPPINGS = {
    "RemacriOnnxUpscaleNode": RemacriOnnxUpscaleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemacriOnnxUpscaleNode": "Remacri ONNX Upscale",
}

__all__ = [
    "RemacriOnnxUpscaleNode",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
]
