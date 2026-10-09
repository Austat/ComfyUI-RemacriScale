# file: ComfyUI/custom_nodes/ComfyUI-RemacriScale/upscale_node_optimized.py
"""GPU-optimized ONNX/TensorRT upscaler for ComfyUI.

Main changes compared with the original implementation:
- CUDA/TensorRT input is bound directly from a PyTorch CUDA tensor.
- ONNX output is kept on the GPU through I/O Binding when the installed
  ONNX Runtime exposes DLPack support.
- Resize, NaN handling and clipping are vectorized in PyTorch.
- ONNX metadata and inference sessions are cached.
- CPU/ROCm and older ONNX Runtime versions retain a safe NumPy fallback.
- Optional timing diagnostics distinguish preparation, inference and
  post-processing time.
"""

import json
import os
import shutil
import time
from typing import Dict, Optional, Tuple

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from tqdm import tqdm

import folder_paths


class RemacriOnnxUpscaleNode:
    """ONNX-based ComfyUI upscaler with GPU I/O binding and true batching."""

    RUNTIME_VERSION_FILE = "./trt_cache_metadata/runtime_versions.json"
    TRT_ENGINE_CACHE_PATH = "./trt_engine_cache"
    TRT_TIMING_CACHE_PATH = "./trt_timing_cache"

    _session = None
    _session_key = None
    _active_provider = None
    _metadata_cache: Dict[str, Tuple[str, Tuple, str, Tuple]] = {}

    @classmethod
    def _check_runtime_versions_and_invalidate_cache(cls):
        current = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "onnxruntime": ort.__version__,
        }
        metadata_dir = os.path.dirname(cls.RUNTIME_VERSION_FILE)
        if metadata_dir:
            os.makedirs(metadata_dir, exist_ok=True)

        previous = None
        if os.path.exists(cls.RUNTIME_VERSION_FILE):
            try:
                with open(cls.RUNTIME_VERSION_FILE, "r", encoding="utf-8") as handle:
                    previous = json.load(handle)
            except (OSError, ValueError, TypeError):
                previous = None

        if previous == current:
            return

        print("[RemacriOnnxUpscale] Runtime versions changed; invalidating TensorRT caches.")
        print(f"  Torch: {previous.get('torch') if previous else None} -> {current['torch']}")
        print(f"  CUDA: {previous.get('cuda') if previous else None} -> {current['cuda']}")
        print(
            "  ONNX Runtime: "
            f"{previous.get('onnxruntime') if previous else None} -> {current['onnxruntime']}"
        )

        for cache_path in (cls.TRT_TIMING_CACHE_PATH, cls.TRT_ENGINE_CACHE_PATH):
            if os.path.isdir(cache_path):
                try:
                    shutil.rmtree(cache_path)
                    print(f"[RemacriOnnxUpscale] Deleted cache directory: {cache_path}")
                except OSError as exc:
                    print(f"[RemacriOnnxUpscale] Failed to delete {cache_path}: {exc}")

        try:
            with open(cls.RUNTIME_VERSION_FILE, "w", encoding="utf-8") as handle:
                json.dump(current, handle, indent=2)
        except OSError as exc:
            print(f"[RemacriOnnxUpscale] Failed to write runtime metadata: {exc}")

        cls._session = None
        cls._session_key = None
        cls._active_provider = None
        cls._metadata_cache.clear()

    @classmethod
    def INPUT_TYPES(cls):
        files = []
        for directory in folder_paths.get_folder_paths("upscale_models"):
            if os.path.isdir(directory):
                for filename in os.listdir(directory):
                    if filename.lower().endswith(".onnx") and filename not in files:
                        files.append(filename)
        files.sort()
        if not files:
            files = ["(no .onnx models found)"]

        return {
            "required": {
                "image": ("IMAGE",),
                "model_file": (files,),
                "provider": ([
                    "TensorrtExecutionProvider",
                    "CUDAExecutionProvider",
                    "ROCmExecutionProvider",
                    "CPUExecutionProvider",
                ],),
                "final_resolution": (["hd", "fhd", "2k", "4k", "8k", "no downscaling"],),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                "diagnostics": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upsampled",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscale"
    OUTPUT_NODE = True

    @staticmethod
    def _find_model(model_file):
        for directory in folder_paths.get_folder_paths("upscale_models"):
            model_path = os.path.join(directory, model_file)
            if os.path.isfile(model_path):
                return model_path
        raise FileNotFoundError(f"Model '{model_file}' not found.")

    @classmethod
    def _model_metadata(cls, model_path):
        cache_key = os.path.abspath(model_path)
        cached = cls._metadata_cache.get(cache_key)
        if cached is not None:
            return cached

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        metadata_session = ort.InferenceSession(
            model_path,
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        model_input = metadata_session.get_inputs()[0]
        model_output = metadata_session.get_outputs()[0]
        metadata = (
            model_input.name,
            tuple(model_input.shape),
            model_output.name,
            tuple(model_output.shape),
        )
        cls._metadata_cache[cache_key] = metadata
        return metadata

    @classmethod
    def _validate_model_input(cls, model_path, requested_batch_size, channels, height, width):
        input_name, input_shape, output_name, output_shape = cls._model_metadata(model_path)
        if len(input_shape) != 4:
            raise RuntimeError(f"Expected a four-dimensional NCHW model input, got {input_shape!r}.")

        fixed_batch = input_shape[0] if isinstance(input_shape[0], int) else None
        if fixed_batch is not None and fixed_batch != requested_batch_size:
            raise RuntimeError(
                f"The ONNX model has fixed batch={fixed_batch}, but batch_size={requested_batch_size}. "
                "Export the model with a dynamic batch dimension for true batching."
            )

        expected = (channels, height, width)
        axis_names = {1: "channels", 2: "height", 3: "width"}
        for axis, (dimension, actual) in enumerate(zip(input_shape[1:], expected), start=1):
            if isinstance(dimension, int) and dimension != actual:
                raise RuntimeError(
                    f"The model requires {axis_names[axis]}={dimension}, got {actual}."
                )
        return input_name, input_shape, output_name, output_shape

    @staticmethod
    def _cuda_stream_pointer() -> Optional[int]:
        if not torch.cuda.is_available():
            return None
        try:
            return int(torch.cuda.current_stream().cuda_stream)
        except Exception:
            return None

    @classmethod
    def _try_create_session(
        cls, model_path, provider, input_name, batch_size, channels, height, width,
        timing_cache_path, diagnostics,
    ):
        available = ort.get_available_providers()
        if provider not in available:
            print(f"[RemacriOnnxUpscale] Provider unavailable: {provider}")
            return None

        session_options = ort.SessionOptions()
        cpu_threads = max(1, min(os.cpu_count() or 8, 16))
        session_options.intra_op_num_threads = cpu_threads
        session_options.inter_op_num_threads = 1
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if diagnostics:
            session_options.enable_profiling = True
            session_options.log_severity_level = 1

        stream_ptr = cls._cuda_stream_pointer()
        providers = None

        cuda_options = {
            "device_id": 0,
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "EXHAUSTIVE",
            "do_copy_in_default_stream": True,
        }
        if stream_ptr is not None:
            cuda_options["user_compute_stream"] = str(stream_ptr)

        if provider == "TensorrtExecutionProvider":
            os.makedirs(os.path.dirname(timing_cache_path), exist_ok=True)
            os.makedirs(cls.TRT_ENGINE_CACHE_PATH, exist_ok=True)
            min_shape = f"{input_name}:1x{channels}x{height}x{width}"
            opt_shape = f"{input_name}:{batch_size}x{channels}x{height}x{width}"
            trt_options = {
                "device_id": 0,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cls.TRT_ENGINE_CACHE_PATH,
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": timing_cache_path,
                "trt_fp16_enable": True,
                "trt_int8_enable": False,
                "trt_dla_enable": False,
                "trt_max_workspace_size": 16 * 1024 * 1024 * 1024,
                "trt_profile_min_shapes": min_shape,
                "trt_profile_opt_shapes": opt_shape,
                "trt_profile_max_shapes": opt_shape,
            }
            if stream_ptr is not None:
                trt_options["user_compute_stream"] = str(stream_ptr)
            providers = [
                ("TensorrtExecutionProvider", trt_options),
                ("CUDAExecutionProvider", cuda_options),
                "CPUExecutionProvider",
            ]
        elif provider == "CUDAExecutionProvider":
            providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
        else:
            providers = [provider]

        started = time.perf_counter()
        try:
            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers,
            )
            print(
                f"[RemacriOnnxUpscale] Initialized providers {session.get_providers()} "
                f"for batch range 1-{batch_size} in {time.perf_counter() - started:.2f}s."
            )
            return session
        except Exception as exc:
            print(f"[RemacriOnnxUpscale] Provider {provider} failed: {exc}")
            return None

    @classmethod
    def _load_session(
        cls, model_path, provider, input_name, batch_size, channels, height, width,
        timing_cache_path, diagnostics,
    ):
        session_key = (
            os.path.abspath(model_path), provider, input_name, batch_size,
            channels, height, width, timing_cache_path, bool(diagnostics),
        )
        if cls._session is not None and cls._session_key == session_key:
            return cls._session

        fallback_chain = [provider, "CUDAExecutionProvider", "ROCmExecutionProvider", "CPUExecutionProvider"]
        tried = set()
        for candidate in fallback_chain:
            if candidate in tried:
                continue
            tried.add(candidate)
            session = cls._try_create_session(
                model_path, candidate, input_name, batch_size, channels, height, width,
                timing_cache_path, diagnostics,
            )
            if session is not None:
                cls._session = session
                cls._session_key = session_key
                cls._active_provider = session.get_providers()[0]
                return session
        raise RuntimeError("All ONNX Runtime execution providers failed.")

    @staticmethod
    def _target_size(final_resolution):
        # torch interpolation uses (height, width)
        return {
            "hd": (720, 1280),
            "fhd": (1080, 1920),
            "2k": (1440, 2560),
            "4k": (2160, 3840),
            "8k": (4320, 7680),
        }.get(final_resolution)

    @staticmethod
    def _ortvalue_to_torch(value):
        """Convert a GPU OrtValue to Torch without a host round trip when supported."""
        try:
            if hasattr(value, "__dlpack__"):
                return torch.utils.dlpack.from_dlpack(value)
            if hasattr(value, "to_dlpack"):
                return torch.utils.dlpack.from_dlpack(value.to_dlpack())
        except Exception as exc:
            print(f"[RemacriOnnxUpscale] DLPack conversion failed; using CPU fallback: {exc}")
        return torch.from_numpy(value.numpy())

    @classmethod
    def _run_gpu_iobinding(cls, session, input_name, output_name, input_batch):
        device_id = input_batch.device.index or 0
        binding = session.io_binding()
        binding.bind_input(
            name=input_name,
            device_type="cuda",
            device_id=device_id,
            element_type=np.float32,
            shape=tuple(input_batch.shape),
            buffer_ptr=input_batch.data_ptr(),
        )
        # Let ORT allocate the dynamic output directly on the CUDA device.
        binding.bind_output(output_name, "cuda", device_id)
        session.run_with_iobinding(binding)
        ort_outputs = binding.get_outputs()
        if not ort_outputs:
            raise RuntimeError("The ONNX model returned no outputs.")
        # clone() gives Torch ownership while keeping the copy device-to-device.
        return cls._ortvalue_to_torch(ort_outputs[0]).clone()

    @staticmethod
    def _run_numpy(session, input_name, input_batch):
        np_input = np.ascontiguousarray(
            input_batch.detach().cpu().numpy().astype(np.float32, copy=False)
        )
        outputs = session.run(None, {input_name: np_input})
        if not outputs:
            raise RuntimeError("The ONNX model returned no outputs.")
        return torch.from_numpy(np.asarray(outputs[0]))

    def upscale(
        self, image, model_file, provider, final_resolution, batch_size,
        diagnostics=False, progress=None,
    ):
        model_path = self._find_model(model_file)
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() != 4:
            raise ValueError(f"Expected IMAGE in NHWC format, got shape {tuple(image.shape)}.")

        total, height, width, channels = image.shape
        if total < 1:
            raise ValueError("The input batch is empty.")
        batch_size = max(1, min(int(batch_size), int(total)))

        self._check_runtime_versions_and_invalidate_cache()
        input_name, _, output_name, _ = self._validate_model_input(
            model_path, batch_size, channels, height, width
        )
        timing_cache_path = os.path.join(
            self.TRT_TIMING_CACHE_PATH,
            f"trt_timing_{height}x{width}_b1-{batch_size}.bin",
        )
        session = self._load_session(
            model_path, provider, input_name, batch_size, channels, height, width,
            timing_cache_path, diagnostics,
        )

        use_cuda_binding = (
            torch.cuda.is_available()
            and self._active_provider in ("TensorrtExecutionProvider", "CUDAExecutionProvider")
        )
        target_device = torch.device("cuda", 0) if use_cuda_binding else torch.device("cpu")
        target_size = self._target_size(final_resolution)

        output_batches = []
        processed = 0
        batch_count = (total + batch_size - 1) // batch_size
        timings = {"prepare": 0.0, "inference": 0.0, "post": 0.0}

        print(f"[RemacriOnnxUpscale] Requested provider: {provider}")
        print(f"[RemacriOnnxUpscale] Active providers: {session.get_providers()}")
        print(f"[RemacriOnnxUpscale] Input device: {image.device}; execution device: {target_device}")

        pbar = tqdm(total=total, desc=f"Upscaling batch 1/{batch_count}", unit="image", dynamic_ncols=True)
        try:
            with torch.inference_mode():
                for batch_index, start in enumerate(range(0, total, batch_size), start=1):
                    end = min(start + batch_size, total)

                    t0 = time.perf_counter()
                    input_batch = (
                        image[start:end]
                        .detach()
                        .permute(0, 3, 1, 2)
                        .contiguous()
                        .to(device=target_device, dtype=torch.float32, non_blocking=True)
                    )
                    if use_cuda_binding and diagnostics:
                        torch.cuda.synchronize(target_device)
                    timings["prepare"] += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    if use_cuda_binding:
                        output_batch = self._run_gpu_iobinding(
                            session, input_name, output_name, input_batch
                        )
                        if diagnostics:
                            torch.cuda.synchronize(target_device)
                    else:
                        output_batch = self._run_numpy(session, input_name, input_batch)
                    timings["inference"] += time.perf_counter() - t0

                    if output_batch.ndim != 4:
                        raise RuntimeError(
                            f"Expected four-dimensional NCHW output, got {tuple(output_batch.shape)}."
                        )
                    if output_batch.shape[0] != end - start:
                        raise RuntimeError(
                            "The ONNX output batch size does not match the input batch size: "
                            f"{output_batch.shape[0]} != {end - start}."
                        )

                    t0 = time.perf_counter()
                    output_batch = torch.nan_to_num(
                        output_batch, nan=0.0, posinf=1.0, neginf=0.0
                    ).clamp_(0.0, 1.0)
                    if target_size is not None and tuple(output_batch.shape[-2:]) != target_size:
                        output_batch = F.interpolate(output_batch, size=target_size, mode="area")
                    output_batch = output_batch.permute(0, 2, 3, 1).contiguous()
                    output_batches.append(output_batch)
                    if use_cuda_binding and diagnostics:
                        torch.cuda.synchronize(target_device)
                    timings["post"] += time.perf_counter() - t0

                    completed = end - start
                    processed += completed
                    pbar.update(completed)
                    pbar.set_description(
                        f"Upscaling batch {batch_index}/{batch_count} ({start + 1}-{end}/{total})"
                    )
                    if progress is not None:
                        progress(int(processed / total * 100))
        finally:
            pbar.close()

        output = torch.cat(output_batches, dim=0)
        if diagnostics:
            total_time = sum(timings.values())
            rate = total / total_time if total_time > 0 else 0.0
            print(
                "[RemacriOnnxUpscale] Timing: "
                f"prepare={timings['prepare']:.3f}s, "
                f"inference={timings['inference']:.3f}s, "
                f"post={timings['post']:.3f}s, "
                f"measured_total={total_time:.3f}s, rate={rate:.2f} images/s"
            )

        return (output,)


NODE_CLASS_MAPPINGS = {
    "RemacriOnnxUpscale": RemacriOnnxUpscaleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemacriOnnxUpscale": "Remacri ONNX Upscale (GPU I/O Binding)",
}
