# file: ComfyUI/custom_nodes/ComfyUI-RemacriScale/remacri_node.py

import json
import os
import shutil
import time

import cv2
import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm

import folder_paths


class RemacriOnnxUpscaleNode:
    """ONNX-based ComfyUI upscaler with true batched inference."""

    RUNTIME_VERSION_FILE = "./trt_cache_metadata/runtime_versions.json"

    _session = None
    _session_key = None
    _active_provider = None

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

        print("[RemacriOnnxUpscale] Runtime versions changed.")
        print(f"  Torch:       {previous.get('torch') if previous else None} -> {current['torch']}")
        print(f"  CUDA:        {previous.get('cuda') if previous else None} -> {current['cuda']}")
        print(
            "  ONNX Runtime: "
            f"{previous.get('onnxruntime') if previous else None} -> {current['onnxruntime']}"
        )
        print("[RemacriOnnxUpscale] Invalidating TensorRT caches...")

        for cache_path in ("./trt_timing_cache", "./trt_engine_cache"):
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
                # Number of input images passed to one session.run() call.
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 64, "step": 1},
                ),
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

    @staticmethod
    def _shape_is_dynamic(value):
        return value is None or isinstance(value, str)

    @classmethod
    def _validate_model_input(cls, model_path, requested_batch_size, channels, height, width):
        # Read model metadata without committing the execution session to a GPU provider.
        metadata_session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )
        model_input = metadata_session.get_inputs()[0]
        shape = model_input.shape

        if len(shape) != 4:
            raise RuntimeError(
                f"Expected a four-dimensional NCHW model input, got {shape!r}."
            )

        fixed_batch = shape[0] if isinstance(shape[0], int) else None
        if fixed_batch is not None and fixed_batch != requested_batch_size:
            raise RuntimeError(
                "The ONNX model has a fixed batch dimension "
                f"({fixed_batch}), but batch_size is {requested_batch_size}. "
                "Export the model with a dynamic batch dimension to use true batching."
            )

        expected = (channels, height, width)
        for axis, (dimension, actual) in enumerate(zip(shape[1:], expected), start=1):
            if isinstance(dimension, int) and dimension != actual:
                axis_names = {1: "channels", 2: "height", 3: "width"}
                raise RuntimeError(
                    f"The model requires {axis_names[axis]}={dimension}, got {actual}."
                )

        return model_input.name, shape

    @classmethod
    def _try_create_session(
        cls,
        model_path,
        provider,
        input_name,
        batch_size,
        channels,
        height,
        width,
        timing_cache_path,
    ):
        available = ort.get_available_providers()
        if provider not in available:
            print(f"[RemacriOnnxUpscale] Provider unavailable: {provider}")
            return None

        if provider == "TensorrtExecutionProvider":
            try:
                free_vram, _ = torch.cuda.mem_get_info()
                print(
                    "[RemacriOnnxUpscale] Free CUDA VRAM before TensorRT session: "
                    f"{free_vram / (1024 ** 3):.2f} GB"
                )
            except Exception as exc:
                print(f"[RemacriOnnxUpscale] CUDA VRAM query failed: {exc}")

        session_options = ort.SessionOptions()
        cpu_threads = os.cpu_count() or 8
        session_options.intra_op_num_threads = cpu_threads
        session_options.inter_op_num_threads = cpu_threads
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        providers = None
        if provider == "TensorrtExecutionProvider":
            os.makedirs(os.path.dirname(timing_cache_path), exist_ok=True)
            os.makedirs("./trt_engine_cache", exist_ok=True)

            # A dynamic TensorRT optimization profile allows all real batch sizes
            # from 1 to the user-selected maximum. The last, smaller batch therefore
            # uses the same engine instead of being padded or processed image by image.
            min_shape = f"{input_name}:1x{channels}x{height}x{width}"
            opt_shape = f"{input_name}:{batch_size}x{channels}x{height}x{width}"
            max_shape = opt_shape

            trt_options = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "./trt_engine_cache",
                "trt_timing_cache_enable": True,
                "trt_timing_cache_path": timing_cache_path,
                "trt_fp16_enable": True,
                "trt_int8_enable": False,
                "trt_dla_enable": False,
                "trt_max_workspace_size": 16 * 1024 * 1024 * 1024,
                "trt_profile_min_shapes": min_shape,
                "trt_profile_opt_shapes": opt_shape,
                "trt_profile_max_shapes": max_shape,
            }
            providers = [
                ("TensorrtExecutionProvider", trt_options),
                "CUDAExecutionProvider",
            ]
        else:
            providers = [provider]

        started = time.time()
        try:
            session = ort.InferenceSession(
                model_path,
                sess_options=session_options,
                providers=providers,
            )
            active = session.get_providers()[0] if session.get_providers() else provider
            print(
                f"[RemacriOnnxUpscale] Initialized {active} "
                f"for batch range 1-{batch_size} in {time.time() - started:.2f} seconds."
            )
            return session
        except Exception as exc:
            print(f"[RemacriOnnxUpscale] Provider {provider} failed: {exc}")
            return None

    @classmethod
    def _load_session(
        cls,
        model_path,
        provider,
        input_name,
        batch_size,
        channels,
        height,
        width,
        timing_cache_path,
    ):
        session_key = (
            model_path,
            provider,
            input_name,
            batch_size,
            channels,
            height,
            width,
            timing_cache_path,
        )
        if cls._session is not None and cls._session_key == session_key:
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

            session = cls._try_create_session(
                model_path=model_path,
                provider=candidate,
                input_name=input_name,
                batch_size=batch_size,
                channels=channels,
                height=height,
                width=width,
                timing_cache_path=timing_cache_path,
            )
            if session is not None:
                cls._session = session
                cls._session_key = session_key
                cls._active_provider = session.get_providers()[0]
                return session

        raise RuntimeError("All ONNX Runtime execution providers failed.")

    @staticmethod
    def _resize_output(image, final_resolution):
        sizes = {
            "hd": (1280, 720),
            "fhd": (1920, 1080),
            "2k": (2560, 1440),
            "4k": (3840, 2160),
            "8k": (7680, 4320),
        }
        target = sizes.get(final_resolution)
        if target is None:
            return image
        return cv2.resize(image, target, interpolation=cv2.INTER_AREA)

    def upscale(
        self,
        image,
        model_file,
        provider,
        final_resolution,
        batch_size,
        progress=None,
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
        input_name, _ = self._validate_model_input(
            model_path,
            requested_batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
        )

        timing_cache_path = os.path.join(
            "./trt_timing_cache",
            f"trt_timing_{height}x{width}_b1-{batch_size}.bin",
        )
        session = self._load_session(
            model_path=model_path,
            provider=provider,
            input_name=input_name,
            batch_size=batch_size,
            channels=channels,
            height=height,
            width=width,
            timing_cache_path=timing_cache_path,
        )

        output_batches = []
        processed = 0
        number_of_batches = (total + batch_size - 1) // batch_size

        pbar = tqdm(
            total=total,
            desc=f"Upscaling batch 1/{number_of_batches}",
            unit="image",
            dynamic_ncols=True,
        )

        try:
            for batch_index, start in enumerate(range(0, total, batch_size), start=1):
                end = min(start + batch_size, total)

                # ComfyUI IMAGE is float NHWC. Convert the complete slice to NCHW,
                # then perform exactly one ONNX Runtime call for this real batch.
                input_batch = (
                    image[start:end]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                    .transpose(0, 3, 1, 2)
                )
                input_batch = np.ascontiguousarray(input_batch)

                ort_outputs = session.run(None, {input_name: input_batch})
                if not ort_outputs:
                    raise RuntimeError("The ONNX model returned no outputs.")

                output_batch = np.asarray(ort_outputs[0])
                if output_batch.ndim != 4:
                    raise RuntimeError(
                        f"Expected four-dimensional NCHW output, got {output_batch.shape}."
                    )
                if output_batch.shape[0] != end - start:
                    raise RuntimeError(
                        "The ONNX output batch size does not match the input batch size: "
                        f"{output_batch.shape[0]} != {end - start}."
                    )

                output_batch = output_batch.transpose(0, 2, 3, 1)
                processed_images = []
                for output_image in output_batch:
                    output_image = self._resize_output(output_image, final_resolution)
                    output_image = np.nan_to_num(
                        output_image,
                        nan=0.0,
                        posinf=1.0,
                        neginf=0.0,
                    )
                    processed_images.append(np.clip(output_image, 0.0, 1.0))

                output_batches.append(
                    np.stack(processed_images, axis=0).astype(np.float32, copy=False)
                )

                completed = end - start
                processed += completed
                pbar.update(completed)
                pbar.set_description(
                    f"Upscaling batch {batch_index}/{number_of_batches} "
                    f"({start + 1}-{end}/{total})"
                )

                if progress is not None:
                    progress(int(processed / total * 100))
        finally:
            pbar.close()

        output = np.concatenate(output_batches, axis=0)
        return (torch.from_numpy(np.ascontiguousarray(output)).float(),)


NODE_CLASS_MAPPINGS = {
    "RemacriOnnxUpscale": RemacriOnnxUpscaleNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RemacriOnnxUpscale": "Remacri ONNX Upscale (True Batch)",
}
