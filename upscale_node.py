# file: ComfyUI/custom_nodes/ComfyUI-RemacriScale/remacri_node.py

import onnxruntime as ort
import numpy as np
import os
import cv2
import torch
import folder_paths
from tqdm import tqdm


class RemacriOnnxUpscaleNode:
    """
    This class implements a custom ComfyUI node for image upscaling using ONNX models.

    ────────────────────────────────────────────────────────────────────────────────
    HIGH‑LEVEL OVERVIEW
    ────────────────────────────────────────────────────────────────────────────────
    • ComfyUI nodes are Python classes with a specific structure.
    • This node loads an ONNX model (Remacri or similar) and performs image upscaling.
    • ONNX Runtime is used as the inference backend.
    • The node supports multiple providers: TensorRT, CUDA, CPU.
    • The node caches the ONNX Runtime session so the model is not reloaded every time.
    • Images are processed one by one (no batching here, because this version is known
      to work reliably and preserve image quality).
    • The output is returned as a PyTorch tensor in ComfyUI format.
    """

    # Cached ONNX Runtime session (shared across calls)
    _session = None

    # Cached model path and provider to detect when a new session must be created
    _model_path = None
    _provider = None

    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the input fields that appear in the ComfyUI node interface.

        ────────────────────────────────────────────────────────────────────────────
        HOW THIS WORKS IN COMFYUI
        ────────────────────────────────────────────────────────────────────────────
        • ComfyUI inspects this dictionary to build the UI.
        • Each key in "required" becomes a visible input.
        • The tuple values define the type and allowed options.
        • "IMAGE" is a special ComfyUI type representing a tensor image.
        """

        # Collect all .onnx files from all configured "upscale_models" directories.
        # ComfyUI allows multiple model folders, so we scan them all.
        files = []
        for d in folder_paths.get_folder_paths("upscale_models"):
            if os.path.isdir(d):
                for f in os.listdir(d):
                    # Only include ONNX files
                    if f.lower().endswith(".onnx") and f not in files:
                        files.append(f)

        # If no ONNX models were found, show a placeholder
        if not files:
            files = ["(no .onnx models found)"]

        # Supported ONNX Runtime providers
        # NOTE: TensorRT may or may not work depending on the model.
        providers = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]

        return {
            "required": {
                "image": ("IMAGE",),                     # Input image tensor
                "model_file": (files,),                  # ONNX model selection
                "provider": (providers,),                # Execution provider
                "final_resolution": (["hd", "fhd", "no downscaling"],),  # Optional resize
            }
        }

    # Output definition for ComfyUI
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upsampled",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscale"
    OUTPUT_NODE = True

    @classmethod
    def _load_session(cls, model_path, provider):
        """
        Loads (or reuses) an ONNX Runtime session.

        ────────────────────────────────────────────────────────────────────────────
        WHY WE CACHE THE SESSION
        ────────────────────────────────────────────────────────────────────────────
        • Loading an ONNX model is expensive.
        • Creating a TensorRT engine is VERY expensive.
        • ComfyUI may call this node many times in a workflow.
        • Therefore we reuse the session unless:
            - the model file changed
            - the provider changed
        """

        # If session does not exist OR model/provider changed → recreate session
        if cls._session is None or cls._model_path != model_path or cls._provider != provider:

            # Create ONNX Runtime session options
            so = ort.SessionOptions()
            # Enable all graph optimizations (constant folding, fusion, etc.)
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # Default provider list
            providers = [provider]

            # If TensorRT is selected, configure TensorRT‑specific options
            if provider == "TensorrtExecutionProvider":
                trt_options = {
                    # Enable TensorRT engine caching (saves compiled engines to disk)
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "./trt_engine_cache",

                    # Enable timing cache (improves engine build speed)
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": "./trt_timing_cache",

                    # Enable FP16 precision (much faster on modern GPUs)
                    "trt_fp16_enable": True,

                    # Disable INT8 (requires calibration data)
                    "trt_int8_enable": False,

                    # Disable DLA (only useful on Jetson devices)
                    "trt_dla_enable": False,
                    "trt_dla_core": 0,

                    # Limit TensorRT workspace memory (prevents 100+ GB allocations)
                    "trt_max_workspace_size": 2 * 1024 * 1024 * 1024,
                }

                # TensorRT with CUDA fallback
                providers = [
                    ("TensorrtExecutionProvider", trt_options),
                    "CUDAExecutionProvider",
                ]

            # Create the ONNX Runtime inference session
            cls._session = ort.InferenceSession(
                model_path,
                sess_options=so,
                providers=providers
            )

            # Cache the model path and provider
            cls._model_path = model_path
            cls._provider = provider

            # Print debug info so the user knows what backend is actually used
            actual_providers = cls._session.get_providers()
            print(f"[RemacriOnnxUpscale] Requested provider: {provider}")
            print(f"[RemacriOnnxUpscale] Actual providers in use: {actual_providers}")

            if provider == "TensorrtExecutionProvider":
                print("[RemacriOnnxUpscale] TensorRT: engine+timing cache enabled, FP16 enabled, INT8 disabled.")

        return cls._session

    def upscale(self, image, model_file, provider, final_resolution, progress=None):
        """
        Main execution function called by ComfyUI.

        ────────────────────────────────────────────────────────────────────────────
        WHAT THIS FUNCTION DOES
        ────────────────────────────────────────────────────────────────────────────
        1. Locates the ONNX model file.
        2. Ensures the input image has a batch dimension.
        3. Loads (or reuses) the ONNX Runtime session.
        4. Processes each image in the batch individually.
        5. Converts ComfyUI tensor → NumPy → ONNX input format.
        6. Runs inference.
        7. Converts ONNX output → NumPy → ComfyUI tensor.
        8. Optionally resizes to HD/FHD.
        9. Returns the upscaled image batch.
        """

        # Search for the model file in all configured upscale model directories
        model_path = None
        search_dirs = folder_paths.get_folder_paths("upscale_models")

        for d in search_dirs:
            p = os.path.join(d, model_file)
            if os.path.exists(p):
                model_path = p
                break

        # If model was not found, raise an error
        if model_path is None:
            raise FileNotFoundError(
                f"Model '{model_file}' not found in any upscale_models directory:\n" +
                "\n".join(search_dirs)
            )

        # Ensure the image tensor has shape [B, H, W, C]
        # ComfyUI sometimes gives [H, W, C] for single images
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # Load or reuse the ONNX Runtime session
        session = self._load_session(model_path, provider)

        out_batch = []          # List to store output images
        total = image.shape[0]  # Number of images in batch

        # Create a progress bar for ComfyUI
        pbar = tqdm(
            total=100,
            desc=f"Upscaling (Image 1/{total})",
            ncols=100,
            colour="blue",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        # Process each image individually (this version is known to be stable)
        for i in range(total):

            # Convert ComfyUI tensor → NumPy uint8 image
            arr = (image[i].cpu().numpy() * 255).astype(np.uint8)

            # Convert HWC → NCHW and normalize to float32 [0,1]
            inp = arr.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

            # Prepare ONNX Runtime input dictionary
            ort_inputs = {session.get_inputs()[0].name: inp}

            # Run inference
            ort_outs = session.run(None, ort_inputs)

            # Convert output NCHW → HWC
            out = ort_outs[0][0].transpose(1, 2, 0)

            # Optional final resolution downscaling
            if final_resolution == "hd":
                out = cv2.resize(out, (1280, 720), interpolation=cv2.INTER_AREA)
            elif final_resolution == "fhd":
                out = cv2.resize(out, (1920, 1080), interpolation=cv2.INTER_AREA)

            # Clean up numerical issues (NaNs, infs)
            out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)

            # Clamp to valid range
            out = np.clip(out, 0.0, 1.0)

            # Store result
            out_batch.append(out)

            # Update progress bar
            percent = int(((i + 1) / total) * 100)

            if progress is not None:
                progress(percent)

            pbar.update(percent - pbar.n)
            pbar.set_description(f"Upscaling (Image {i+1}/{total})")

        pbar.close()

        # Stack all outputs into a single NumPy array
        out = np.stack(out_batch, axis=0).astype(np.float32)

        # Convert NumPy → PyTorch tensor for ComfyUI
        out_tensor = torch.from_numpy(out).float()

        return (out_tensor,)
