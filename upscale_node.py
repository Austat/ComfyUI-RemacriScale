import json
import os
import shutil
import time

import cv2
import folder_paths
import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm


class RemacriOnnxUpscaleNode:
    """ComfyUI ONNX upscaler with TensorRT/CUDA/ROCm/CPU fallback,
    ONNX Runtime I/O binding, and configurable inference batches.
    """

    RUNTIME_VERSION_FILE = "./trt_cache_metadata/runtime_versions.json"

    _session = None
    _model_path = None
    _requested_provider = None
    _active_provider = None
    _timing_cache_path = None

    @classmethod
    def _check_runtime_versions_and_invalidate_cache(cls, timing_cache_path):
        current = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "onnxruntime": ort.__version__,
        }

        meta_dir = os.path.dirname(cls.RUNTIME_VERSION_FILE)
        if meta_dir:
            os.makedirs(meta_dir, exist_ok=True)

        previous = None
        if os.path.exists(cls.RUNTIME_VERSION_FILE):
            try:
                with open(cls.RUNTIME_VERSION_FILE, "r", encoding="utf-8") as f:
                    previous = json.load(f)
            except (OSError, ValueError, TypeError):
                previous = None

        if previous == current:
            return

        print("[RemacriOnnxUpscale] Runtime versions changed:")
        print(f"  Torch:       {previous.get('torch') if previous else None} -> {current['torch']}")
        print(f"  CUDA:        {previous.get('cuda') if previous else None} -> {current['cuda']}")
        print(
            f"  ONNXRuntime: {previous.get('onnxruntime') if previous else None} "
            f"-> {current['onnxruntime']}"
        )
        print("[RemacriOnnxUpscale] Invalidating TensorRT timing and engine caches...")

        timing_cache_dir = os.path.dirname(timing_cache_path)
        if timing_cache_dir and os.path.isdir(timing_cache_dir):
            try:
                shutil.rmtree(timing_cache_dir)
                print(f"[RemacriOnnxUpscale] Deleted timing cache directory: {timing_cache_dir}")
            except OSError as e:
                print(f"[RemacriOnnxUpscale] Failed to delete timing cache directory: {e}")

        engine_cache_dir = "./trt_engine_cache"
        if os.path.isdir(engine_cache_dir):
            try:
                shutil.rmtree(engine_cache_dir)
                print(f"[RemacriOnnxUpscale] Deleted engine cache directory: {engine_cache_dir}")
            except OSError as e:
                print(f"[RemacriOnnxUpscale] Failed to delete engine cache directory: {e}")

        try:
            with open(cls.RUNTIME_VERSION_FILE, "w", encoding="utf-8") as f:
                json.dump(current, f, indent=2)
        except OSError as e:
            print(f"[RemacriOnnxUpscale] Failed to write version metadata: {e}")

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

        providers = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "ROCmExecutionProvider",
            "CPUExecutionProvider",
        ]
        resolutions = ["hd", "fhd", "2k", "4k", "8k", "no downscaling"]

        return {
            "required": {
                "image": ("IMAGE",),
                "model_file": (files,),
                "provider": (providers,),
                "final_resolution": (resolutions,),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 8, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upsampled",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscale"
    OUTPUT_NODE = True

    @classmethod
    def _try_create_session(cls, model_path, provider, timing_cache_path):
        available = ort.get_available_providers()
        if provider not in available:
            print(f"[RemacriOnnxUpscale] Provider not available: {provider}")
            return None

        if provider == "TensorrtExecutionProvider":
            try:
                free_vram, total_vram = torch.cuda.mem_get_info()
                print(
                    f"[RemacriOnnxUpscale] CUDA VRAM free: "
                    f"{free_vram / (1024 ** 3):.2f}/{total_vram / (1024 ** 3):.2f} GB"
                )
            except Exception as e:
                print(f"[RemacriOnnxUpscale] VRAM check failed, skipping TensorRT: {e}")
                return None

        try:
            session_options = ort.SessionOptions()
            cpu_threads = os.cpu_count() or 8
            session_options.intra_op_num_threads = cpu_threads
            session_options.inter_op_num_threads = cpu_threads
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.enable_mem_pattern = True
            session_options.enable_mem_reuse = True

            timing_build_start = None
            engine_build_start = None
            engine_cache_files_before = set()

            if provider == "TensorrtExecutionProvider":
                timing_cache_dir = os.path.dirname(timing_cache_path)
                if timing_cache_dir:
                    os.makedirs(timing_cache_dir, exist_ok=True)

                engine_cache_dir = "./trt_engine_cache"
                os.makedirs(engine_cache_dir, exist_ok=True)

                if not os.path.exists(timing_cache_path):
                    print("[RemacriOnnxUpscale] Timing cache not found; TensorRT may build tactics.")
                    timing_build_start = time.perf_counter()

                engine_cache_files_before = set(os.listdir(engine_cache_dir))
                engine_build_start = time.perf_counter()

                trt_options = {
                    "device_id": 0,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": engine_cache_dir,
                    "trt_timing_cache_enable": True,
                    "trt_timing_cache_path": timing_cache_path,
                    "trt_fp16_enable": True,
                    "trt_int8_enable": False,
                    "trt_dla_enable": False,
                    "trt_dla_core": 0,
                    "trt_max_workspace_size": 16 * 1024 * 1024 * 1024,
                }
                providers = [
                    ("TensorrtExecutionProvider", trt_options),
                    ("CUDAExecutionProvider", {"device_id": 0}),
                ]
            elif provider == "CUDAExecutionProvider":
                providers = [("CUDAExecutionProvider", {"device_id": 0})]
            elif provider == "ROCmExecutionProvider":
                providers = [("ROCmExecutionProvider", {"device_id": 0})]
            else:
                providers = ["CPUExecutionProvider"]

            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers,
            )

            active = session.get_providers()
            print(f"[RemacriOnnxUpscale] Requested provider: {provider}")
            print(f"[RemacriOnnxUpscale] Session providers: {active}")

            if provider == "TensorrtExecutionProvider":
                if timing_build_start is not None:
                    print(
                        "[RemacriOnnxUpscale] Session/timing-cache initialization took "
                        f"{time.perf_counter() - timing_build_start:.2f} seconds."
                    )

                engine_cache_files_after = set(os.listdir("./trt_engine_cache"))
                new_files = engine_cache_files_after - engine_cache_files_before
                if new_files:
                    print(
                        "[RemacriOnnxUpscale] TensorRT engine cache initialization took "
                        f"{time.perf_counter() - engine_build_start:.2f} seconds."
                    )
                    print(f"[RemacriOnnxUpscale] New engine files: {', '.join(sorted(new_files))}")
                else:
                    print(
                        "[RemacriOnnxUpscale] No new engine file appeared during session creation. "
                        "A dynamic engine may be built on the first inference."
                    )

            return session
        except Exception as e:
            print(f"[RemacriOnnxUpscale] Provider {provider} failed: {e}")
            return None

    @classmethod
    def _load_session(cls, model_path, provider, timing_cache_path):
        if (
            cls._session is not None
            and cls._model_path == model_path
            and cls._requested_provider == provider
            and cls._timing_cache_path == timing_cache_path
        ):
            return cls._session

        fallback_chain = [
            provider,
            "CUDAExecutionProvider",
            "ROCmExecutionProvider",
            "CPUExecutionProvider",
        ]

        tried = set()
        for candidate in fallback_chain:
            if candidate in tried:
                continue
            tried.add(candidate)

            session = cls._try_create_session(model_path, candidate, timing_cache_path)
            if session is not None:
                cls._session = session
                cls._model_path = model_path
                cls._requested_provider = provider
                cls._active_provider = candidate
                cls._timing_cache_path = timing_cache_path
                print(f"[RemacriOnnxUpscale] Using provider: {candidate}")
                return session

        raise RuntimeError("All ONNX Runtime providers failed.")

    @staticmethod
    def _uses_cuda_device(session):
        providers = session.get_providers()
        return (
            "TensorrtExecutionProvider" in providers
            or "CUDAExecutionProvider" in providers
        ) and torch.cuda.is_available()

    @staticmethod
    def _run_with_iobinding(session, batch_nhwc):
        """Run one NHWC float32 batch with ORT I/O binding.

        CUDA/TensorRT input is bound directly to a contiguous CUDA torch tensor.
        The output is allocated by ORT on the execution device and copied to CPU
        only after inference because ComfyUI IMAGE output is assembled on CPU.
        """
        input_meta = session.get_inputs()[0]
        output_meta = session.get_outputs()[0]
        input_name = input_meta.name
        output_name = output_meta.name
        io_binding = session.io_binding()

        if RemacriOnnxUpscaleNode._uses_cuda_device(session):
            input_tensor = (
                batch_nhwc.permute(0, 3, 1, 2)
                .contiguous()
                .to(device="cuda:0", dtype=torch.float32, non_blocking=True)
            )

            io_binding.bind_input(
                name=input_name,
                device_type="cuda",
                device_id=0,
                element_type=np.float32,
                shape=tuple(input_tensor.shape),
                buffer_ptr=input_tensor.data_ptr(),
            )
            io_binding.bind_output(output_name, "cuda", 0)

            session.run_with_iobinding(io_binding)
            io_binding.synchronize_outputs()
            output_nchw = io_binding.copy_outputs_to_cpu()[0]

            # Keep input_tensor alive until inference and output synchronization finish.
            del input_tensor
        else:
            input_nchw = (
                batch_nhwc.permute(0, 3, 1, 2)
                .contiguous()
                .cpu()
                .numpy()
                .astype(np.float32, copy=False)
            )
            input_ortvalue = ort.OrtValue.ortvalue_from_numpy(input_nchw)
            io_binding.bind_ortvalue_input(input_name, input_ortvalue)
            io_binding.bind_output(output_name, "cpu")
            session.run_with_iobinding(io_binding)
            output_nchw = io_binding.copy_outputs_to_cpu()[0]

        return np.asarray(output_nchw).transpose(0, 2, 3, 1)

    @staticmethod
    def _resize_output(output, final_resolution):
        sizes = {
            "hd": (1280, 720),
            "fhd": (1920, 1080),
            "2k": (2560, 1440),
            "4k": (3840, 2160),
            "8k": (7680, 4320),
        }
        size = sizes.get(final_resolution)
        if size is None:
            return output
        return cv2.resize(output, size, interpolation=cv2.INTER_AREA)

    def upscale(self, image, model_file, provider, final_resolution, batch_size=1, progress=None):
        model_path = None
        for directory in folder_paths.get_folder_paths("upscale_models"):
            candidate = os.path.join(directory, model_file)
            if os.path.isfile(candidate):
                model_path = candidate
                break

        if model_path is None:
            raise FileNotFoundError(f"Model '{model_file}' not found.")

        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() != 4:
            raise ValueError(f"Expected IMAGE tensor with 4 dimensions, got {tuple(image.shape)}")

        batch_size = max(1, min(8, int(batch_size)))
        total, height, width, channels = image.shape
        if channels not in (1, 3, 4):
            raise ValueError(f"Unsupported input channel count: {channels}")

        timing_cache_path = os.path.join(
            "./trt_timing_cache",
            f"trt_timing_cache_{height}x{width}_b{batch_size}.bin",
        )
        self._check_runtime_versions_and_invalidate_cache(timing_cache_path)
        session = self._load_session(model_path, provider, timing_cache_path)

        print(
            f"[RemacriOnnxUpscale] Processing {total} image(s), "
            f"requested batch size {batch_size}."
        )

        output_images = []
        pbar = tqdm(
            total=total,
            desc=f"Upscaling (0/{total})",
            ncols=100,
            colour="blue",
            dynamic_ncols=True,
        )

        processed = 0
        try:
            for start in range(0, total, batch_size):
                end = min(start + batch_size, total)
                current = image[start:end].detach()

                # Preserve float input precision. ComfyUI IMAGE values are expected in [0, 1].
                current = current.to(dtype=torch.float32).clamp_(0.0, 1.0)

                try:
                    batch_output = self._run_with_iobinding(session, current)
                except Exception as e:
                    raise RuntimeError(
                        f"ONNX inference failed for batch {start + 1}-{end} "
                        f"with batch size {end - start}. The model may have a fixed batch "
                        f"dimension of 1. Select batch_size=1 or export the ONNX model "
                        f"with a dynamic batch dimension. Original error: {e}"
                    ) from e

                for output in batch_output:
                    output = self._resize_output(output, final_resolution)
                    output = np.nan_to_num(output, nan=0.0, posinf=1.0, neginf=0.0)
                    output = np.clip(output, 0.0, 1.0).astype(np.float32, copy=False)
                    output_images.append(output)

                done = end - start
                processed += done
                pbar.update(done)
                pbar.set_description(f"Upscaling ({processed}/{total})")
                if progress is not None:
                    progress(int(processed / total * 100))
        finally:
            pbar.close()

        output_array = np.stack(output_images, axis=0).astype(np.float32, copy=False)
        return (torch.from_numpy(output_array),)
