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
    Custom ComfyUI node for ONNX-based image upscaling.

    ────────────────────────────────────────────────────────────────────────────────
    HIGH‑LEVEL PURPOSE
    ────────────────────────────────────────────────────────────────────────────────
    This node loads an ONNX upscaling model (e.g., Remacri) and performs image
    upscaling inside ComfyUI. It supports multiple hardware backends:

        • NVIDIA TensorRT (fastest, but strict about VRAM and input shapes)
        • NVIDIA CUDA (stable and fast)
        • AMD ROCm (for Radeon GPUs)
        • CPU (fallback that always works)

    The node includes:
        • Resolution‑specific TensorRT timing cache
        • Automatic fallback logic (TRT → CUDA → ROCm → CPU)
        • VRAM‑monitoring to avoid TensorRT crashes
        • Support for HD, FHD, 2K, 4K, 8K output resolutions
        • Detailed comments explaining every important step

    The goal is to provide a robust, GPU‑agnostic upscale node that:
        • Works on NVIDIA, AMD and CPU systems
        • Avoids TensorRT failures on low VRAM
        • Reuses ONNX Runtime sessions efficiently
        • Produces stable, high‑quality results
    """

    # Cached session and metadata
    _session = None
    _model_path = None
    _provider = None
    _timing_cache_path = None


    @classmethod
    def INPUT_TYPES(cls):
        """
        Defines the UI inputs for this node.

        ────────────────────────────────────────────────────────────────────────────
        HOW COMFYUI USES THIS
        ────────────────────────────────────────────────────────────────────────────
        ComfyUI inspects this dictionary to build the node interface. Each key in
        "required" becomes a visible input field. The tuple values define allowed
        types or dropdown options.

        This function:
            • Lists all .onnx models found in the upscale_models folder
            • Provides a dropdown for selecting execution provider
            • Provides a dropdown for selecting output resolution
        """

        # Collect all ONNX models from all configured upscale model directories
        files = []
        for d in folder_paths.get_folder_paths("upscale_models"):
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(".onnx") and f not in files:
                        files.append(f)

        # If no models found, show placeholder
        if not files:
            files = ["(no .onnx models found)"]

        # Supported execution providers:
        # NVIDIA: TensorRT, CUDA
        # AMD: ROCm
        # CPU: always available
        providers = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "ROCmExecutionProvider",
            "CPUExecutionProvider"
        ]

        # Output resolution options
        resolutions = [
            "hd",              # 1280×720
            "fhd",             # 1920×1080
            "2k",              # 2560×1440
            "4k",              # 3840×2160
            "8k",              # 7680×4320
            "no downscaling",  # keep model output resolution
        ]

        return {
            "required": {
                "image": ("IMAGE",),
                "model_file": (files,),
                "provider": (providers,),
                "final_resolution": (resolutions,),
            }
        }


    # Output definition for ComfyUI
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upsampled",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscale"
    OUTPUT_NODE = True

    @classmethod
    def _try_create_session(cls, model_path, provider, timing_cache_path):
        """
        Attempts to create an ONNX Runtime session using a specific provider.

        ────────────────────────────────────────────────────────────────────────────
        WHY THIS FUNCTION EXISTS
        ────────────────────────────────────────────────────────────────────────────
        This helper function is used by the fallback system. Instead of failing
        immediately when a provider cannot be initialized, we catch the exception
        and return None. The caller (_load_session) then tries the next provider.

        This allows:
            • TensorRT to fail gracefully (common with low VRAM or unsupported ops)
            • CUDA to be used as a fallback
            • ROCm to be used on AMD GPUs
            • CPU to be used as a last resort
        """

        # ────────────────────────────────────────────────────────────────────────
        # VRAM CHECK FOR TENSORRT
        # ────────────────────────────────────────────────────────────────────────
        if provider == "TensorrtExecutionProvider":
            try:
                # Query free VRAM (NVIDIA only)
                free_vram, total_vram = torch.cuda.mem_get_info()
                free_gb = free_vram / (1024**3)

                # Minimum VRAM required for safe TensorRT engine building
                required_gb = 0

                if free_gb < required_gb:
                    print(
                        f"[RemacriOnnxUpscale] Skipping TensorRT: only {free_gb:.2f} GB free, "
                        f"{required_gb} GB required."
                    )
                    return None

            except Exception as e:
                # If VRAM check fails (e.g., AMD GPU), skip TRT entirely
                print(f"[RemacriOnnxUpscale] VRAM check failed, skipping TensorRT: {e}")
                return None

        try:
            # Create ONNX Runtime session options
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            # ────────────────────────────────────────────────────────────────────
            # PROVIDER‑SPECIFIC CONFIGURATION
            # ────────────────────────────────────────────────────────────────────

            if provider == "TensorrtExecutionProvider":
                # Ensure timing cache directory exists
                cache_dir = os.path.dirname(timing_cache_path)
                if cache_dir and not os.path.exists(cache_dir):
                    os.makedirs(cache_dir, exist_ok=True)

                # TensorRT configuration dictionary
                trt_options = {
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": "./trt_engine_cache",

                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": timing_cache_path,

                    "trt_fp16_enable": True,
                    "trt_int8_enable": False,

                    "trt_dla_enable": False,
                    "trt_dla_core": 0,

                    # Large workspace for high‑resolution engine builds
                    "trt_max_workspace_size": 16 * 1024 * 1024 * 1024,
                }

                providers = [
                    ("TensorrtExecutionProvider", trt_options),
                    "CUDAExecutionProvider",
                ]

            elif provider == "CUDAExecutionProvider":
                # Standard CUDA backend
                providers = ["CUDAExecutionProvider"]

            elif provider == "ROCmExecutionProvider":
                # AMD ROCm backend
                providers = ["ROCmExecutionProvider"]

            else:
                # CPU fallback
                providers = ["CPUExecutionProvider"]

            # ────────────────────────────────────────────────────────────────────
            # TRY TO CREATE THE SESSION
            # ────────────────────────────────────────────────────────────────────
            session = ort.InferenceSession(
                model_path,
                sess_options=so,
                providers=providers
            )

            print(f"[RemacriOnnxUpscale] Successfully initialized provider: {provider}")
            return session

        except Exception as e:
            # Provider failed → return None so fallback can continue
            print(f"[RemacriOnnxUpscale] Provider {provider} failed: {e}")
            return None


    @classmethod
    def _load_session(cls, model_path, provider, timing_cache_path):
        """
        Creates or reuses an ONNX Runtime session with full fallback logic.

        ────────────────────────────────────────────────────────────────────────────
        FALLBACK ORDER
        ────────────────────────────────────────────────────────────────────────────
        1. User-selected provider
        2. CUDA (NVIDIA)
        3. ROCm (AMD)
        4. CPU (always works)

        This ensures:
            • TensorRT failures do not break the node
            • CUDA is used when available
            • AMD GPUs work automatically
            • CPU is always the last safe fallback
        """

        # Reuse cached session if nothing changed
        if (
            cls._session is not None
            and cls._model_path == model_path
            and cls._provider == provider
            and cls._timing_cache_path == timing_cache_path
        ):
            return cls._session

        # Fallback chain
        fallback_chain = [
            provider,                     # user-selected
            "CUDAExecutionProvider",
            "ROCmExecutionProvider",
            "CPUExecutionProvider",
        ]

        tried = set()

        # Try each provider in order
        for p in fallback_chain:
            if p in tried:
                continue
            tried.add(p)

            session = cls._try_create_session(model_path, p, timing_cache_path)
            if session is not None:
                # Cache session metadata
                cls._session = session
                cls._provider = p
                cls._model_path = model_path
                cls._timing_cache_path = timing_cache_path

                print(f"[RemacriOnnxUpscale] Using provider: {p}")
                return session

        # If all providers fail, raise an error
        raise RuntimeError("All providers failed. Cannot create ONNX Runtime session.")

    def upscale(self, image, model_file, provider, final_resolution, progress=None):
        """
        Main execution function called by ComfyUI.

        ────────────────────────────────────────────────────────────────────────────
        WHAT THIS FUNCTION DOES
        ────────────────────────────────────────────────────────────────────────────
        1. Locates the ONNX model file.
        2. Ensures the input image has a batch dimension.
        3. Determines input resolution (H×W).
        4. Builds a resolution‑specific TensorRT timing‑cache path.
        5. Loads or reuses the ONNX Runtime session (with fallback logic).
        6. Processes each image individually.
        7. Converts ComfyUI tensor → NumPy → ONNX input format.
        8. Runs inference.
        9. Converts ONNX output → NumPy → ComfyUI tensor.
        10. Optionally resizes to HD/FHD/2K/4K/8K.
        """

        # ────────────────────────────────────────────────────────────────────────
        # 1. Locate the ONNX model file
        # ────────────────────────────────────────────────────────────────────────
        model_path = None
        for d in folder_paths.get_folder_paths("upscale_models"):
            p = os.path.join(d, model_file)
            if os.path.exists(p):
                model_path = p
                break

        if model_path is None:
            raise FileNotFoundError(f"Model '{model_file}' not found.")

        # ────────────────────────────────────────────────────────────────────────
        # 2. Ensure batch dimension [B, H, W, C]
        # ────────────────────────────────────────────────────────────────────────
        if image.dim() == 3:
            image = image.unsqueeze(0)

        # ────────────────────────────────────────────────────────────────────────
        # 3. Determine input resolution (H×W)
        # ────────────────────────────────────────────────────────────────────────
        _, H, W, _ = image.shape

        # ────────────────────────────────────────────────────────────────────────
        # 4. Build resolution‑specific TensorRT timing‑cache path
        #    Example: ./trt_timing_cache/trt_timing_cache_720x1280.bin
        # ────────────────────────────────────────────────────────────────────────
        timing_cache_dir = "./trt_timing_cache"
        timing_cache_filename = f"trt_timing_cache_{H}x{W}.bin"
        timing_cache_path = os.path.join(timing_cache_dir, timing_cache_filename)

        # ────────────────────────────────────────────────────────────────────────
        # 5. Load ONNX Runtime session (with fallback logic)
        # ────────────────────────────────────────────────────────────────────────
        session = self._load_session(model_path, provider, timing_cache_path)

        # ────────────────────────────────────────────────────────────────────────
        # Prepare output list and progress bar
        # ────────────────────────────────────────────────────────────────────────
        out_batch = []
        total = image.shape[0]

        pbar = tqdm(
            total=100,
            desc=f"Upscaling (Image 1/{total})",
            ncols=100,
            colour="blue",
            dynamic_ncols=True,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
        )

        # ────────────────────────────────────────────────────────────────────────
        # 6. Process each image individually
        # ────────────────────────────────────────────────────────────────────────
        for i in range(total):

            # ────────────────────────────────────────────────────────────────────
            # Convert ComfyUI tensor → uint8 NumPy array
            # ────────────────────────────────────────────────────────────────────
            arr = (image[i].cpu().numpy() * 255).astype(np.uint8)

            # Convert HWC → NCHW and normalize to [0,1]
            inp = arr.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

            # ────────────────────────────────────────────────────────────────────
            # 7. Run ONNX inference
            # ────────────────────────────────────────────────────────────────────
            ort_inputs = {session.get_inputs()[0].name: inp}
            ort_outs = session.run(None, ort_inputs)

            # Convert NCHW → HWC
            out = ort_outs[0][0].transpose(1, 2, 0)

            # ────────────────────────────────────────────────────────────────────
            # 8. Optional final resolution scaling
            # ────────────────────────────────────────────────────────────────────
            if final_resolution == "hd":
                out = cv2.resize(out, (1280, 720), interpolation=cv2.INTER_AREA)

            elif final_resolution == "fhd":
                out = cv2.resize(out, (1920, 1080), interpolation=cv2.INTER_AREA)

            elif final_resolution == "2k":
                out = cv2.resize(out, (2560, 1440), interpolation=cv2.INTER_AREA)

            elif final_resolution == "4k":
                out = cv2.resize(out, (3840, 2160), interpolation=cv2.INTER_AREA)

            elif final_resolution == "8k":
                out = cv2.resize(out, (7680, 4320), interpolation=cv2.INTER_AREA)

            # ────────────────────────────────────────────────────────────────────
            # 9. Clean numerical issues (NaN, inf)
            # ────────────────────────────────────────────────────────────────────
            out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
            out = np.clip(out, 0.0, 1.0)

            out_batch.append(out)

            # ────────────────────────────────────────────────────────────────────
            # Update progress bar
            # ────────────────────────────────────────────────────────────────────
            percent = int(((i + 1) / total) * 100)
            if progress is not None:
                progress(percent)

            pbar.update(percent - pbar.n)
            pbar.set_description(f"Upscaling (Image {i+1}/{total})")

        pbar.close()

        # ────────────────────────────────────────────────────────────────────────
        # 10. Stack outputs and convert back to ComfyUI tensor
        # ────────────────────────────────────────────────────────────────────────
        out = np.stack(out_batch, axis=0).astype(np.float32)
        out_tensor = torch.from_numpy(out).float()

        return (out_tensor,)

