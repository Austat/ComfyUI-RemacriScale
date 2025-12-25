# ComfyUI-RemacriScale
Using Remacri upscaler, upscale a video using one of three providers supported by onnx and then downscale.

Supported methods are:

           "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider"

<img width="315" height="128" alt="image" src="https://github.com/user-attachments/assets/8d4fc74a-b646-4094-b2d7-3214f261f50e" />


Installation

Method 1: Clone the Repository Navigate to your ComfyUI custom_nodes directory. Run:

git clone https://github.com/Austat/ComfyUI-RemacriScale

cd ComfyUI-RemacriScale

pip install -r requirements.txt

Download needed onnx - files to your ComfyUI or custom models/upscale_models/ - folder.

Restart ComfyUI.

First upscaling run will take longer time as each used resolution needs it's own TensorRT engine. Subsident runs are considerably faster as they use the previously created timing cache.

