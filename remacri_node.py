# file: ComfyUI/custom_nodes/ComfyUI-RemacriScale/remacri_node.py

# ---------------------------------------------------------------------------
# Import dependencies
# ---------------------------------------------------------------------------

# ONNX Runtime is the inference engine used to run ONNX models.
# It supports multiple execution providers (CPU, CUDA, TensorRT).
import onnxruntime as ort

# NumPy is used for numerical array manipulation.
# We rely on it to reshape, normalize, and stack image data.
import numpy as np

# os is used for filesystem operations such as checking directories and building paths.
import os

# OpenCV (cv2) is used for image resizing.
# After upscaling, we can optionally downscale to HD/FHD resolutions.
import cv2

# PyTorch is imported because ComfyUI expects images to be torch.FloatTensor objects.
# This ensures compatibility with the rest of the pipeline.
import torch


class RemacriOnnxUpscaleNode:
    """
    Custom ComfyUI node for image upscaling using ONNX models.

    Key features:
    - Dropdown for selecting ONNX model from models/upscale_models/ folder.
    - Dropdown for selecting execution provider (TensorRT, CUDA, CPU).
    - Optional downscale to HD (1280x720) or FHD (1920x1080).
    - Reports progress (0–100%) back to ComfyUI during batch processing.
    """

    # -----------------------------------------------------------------------
    # Class-level cache variables
    # -----------------------------------------------------------------------
    # These are used to avoid reloading the ONNX model session every time.
    _session = None       # Cached ONNX Runtime session.
    _model_path = None    # Path of the currently loaded model.
    _provider = None      # Name of the currently selected provider.

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the inputs that appear in ComfyUI's interface.

        Steps:
        - Scan models/upscale_models/ for .onnx files and list them in a dropdown.
        - Provide a dropdown for execution provider selection.
        - Provide a dropdown for final resolution selection.
        """

        # Path to the model directory.
        model_dir = os.path.join("models", "upscale_models")

        # Collect all .onnx files in the directory.
        files = []
        if os.path.isdir(model_dir):
            files = [f for f in os.listdir(model_dir) if f.endswith(".onnx")]

        # If no models are found, insert a placeholder entry.
        if not files:
            files = ["(no .onnx models found)"]

        # Available execution providers for the dropdown.
        providers = ["TensorRTExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

        # Return the input specification dictionary for ComfyUI.
        return {
            "required": {
                "image": ("IMAGE",),                # Input image tensor.
                "model_file": (files,),             # Dropdown list of ONNX models.
                "provider": (providers,),           # Dropdown list of providers.
                "final_resolution": (["none", "hd", "fhd"],),  # Dropdown for optional downscale.
            }
        }

    # -----------------------------------------------------------------------
    # Output specification
    # -----------------------------------------------------------------------
    RETURN_TYPES = ("IMAGE",)       # Output is an image.
    RETURN_NAMES = ("upsampled",)   # Label for the output.
    FUNCTION = "upscale"            # Function executed when node runs.
    CATEGORY = "image/upscale"      # Category in ComfyUI's node menu.

    # This flag tells ComfyUI that the node can report progress.
    OUTPUT_NODE = True

    @classmethod
    def _load_session(cls, model_path, provider):
        """
        Loads or reuses an ONNX Runtime session.

        Logic:
        - If no session is cached, or if the model/provider changed, create a new session.
        - Cache the session, model path, and provider for reuse.
        - Print which provider is being used for debugging.
        """

        if cls._session is None or cls._model_path != model_path or cls._provider != provider:
            cls._session = ort.InferenceSession(model_path, providers=[provider])
            cls._model_path = model_path
            cls._provider = provider
            print(f"[RemacriOnnxUpscale] Using provider: {provider}")
        return cls._session

    def upscale(self, image, model_file, provider, final_resolution, progress=None):
        """
        Main execution function.

        Steps:
        1. Build full path to selected ONNX model.
        2. Verify model exists.
        3. Load or reuse ONNX Runtime session.
        4. Iterate over batch of images:
           - Convert tensor to NumPy array (uint8).
           - Normalize and reshape to NCHW float32 format.
           - Run inference with ONNX Runtime.
           - Convert output back to HWC format.
           - Optionally downscale to HD/FHD.
           - Append to output batch.
           - Report progress percentage to ComfyUI.
        5. Stack outputs, convert to torch.FloatTensor, and return.
        """

        # Construct full path to the model file.
        model_path = os.path.join("models", "upscale_models", model_file)

        # Ensure the model file exists.
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load or reuse ONNX Runtime session.
        session = self._load_session(model_path, provider)

        # Prepare list to collect output images.
        out_batch = []

        # Total number of images in batch.
        total = image.shape[0]

        # Process each image in the batch.
        for i in range(total):
            # Convert torch tensor to NumPy array in uint8 format (0–255).
            arr = (image[i].cpu().numpy() * 255).astype(np.uint8)

            # Rearrange dimensions from HWC to NCHW, add batch dimension, normalize to 0–1 float32.
            inp = arr.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

            # Prepare ONNX Runtime input dictionary.
            ort_inputs = {session.get_inputs()[0].name: inp}

            # Run inference with ONNX model.
            ort_outs = session.run(None, ort_inputs)

            # Convert output back from NCHW to HWC format.
            out = ort_outs[0][0].transpose(1, 2, 0)

            # Optional downscale step.
            if final_resolution == "hd":
                out = cv2.resize(out, (1280, 720), interpolation=cv2.INTER_AREA)
            elif final_resolution == "fhd":
                out = cv2.resize(out, (1920, 1080), interpolation=cv2.INTER_AREA)

            # Append processed image to batch.
            out_batch.append(out)

            # Report progress to ComfyUI (if callback provided).
            if progress is not None:
                percent = int(((i + 1) / total) * 100)
                progress(percent)  # This updates the UI progress bar.
                print(f"[RemacriOnnxUpscale] Progress: {percent}%")  # Console log for debugging.

        # Stack all processed images into one NumPy array.
        out = np.stack(out_batch, axis=0).astype(np.float32)

        # Convert NumPy array to torch.FloatTensor (ComfyUI expects torch tensors).
        out_tensor = torch.from_numpy(out).float()

        # Return output as tuple (ComfyUI requires tuple outputs).
        return (out_tensor,)
