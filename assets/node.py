import os
import time
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

import folder_paths

from .cache import RuntimeCacheMixin
from .inference import InferenceMixin
from .model import ModelMixin
from .session import SessionMixin


class RemacriOnnxUpscaleNode(
    RuntimeCacheMixin,
    ModelMixin,
    SessionMixin,
    InferenceMixin,
):
    """ONNX-based ComfyUI upscaler with GPU I/O binding and true batching."""

    RUNTIME_VERSION_FILE = "./trt_cache_metadata/runtime_versions.json"
    TRT_ENGINE_CACHE_PATH = "./trt_engine_cache"
    TRT_TIMING_CACHE_PATH = "./trt_timing_cache"

    _session = None
    _session_key = None
    _active_provider = None
    _metadata_cache: Dict[str, Tuple[str, Tuple, str, Tuple]] = {}

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
                    "final_resolution": ([
                        "hd", "fhd", "2k", "4k", "8k",
                        "512x512", "768x768", "1024x1024", "2048x2048", "4096x4096",
                        "no downscaling",
                    ],),
                    "batch_size": ("INT", {"default": 4, "min": 1, "max": 64, "step": 1}),
                    "diagnostics": ("BOOLEAN", {"default": False}),
                }
            }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upsampled",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscale"
    OUTPUT_NODE = True

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
