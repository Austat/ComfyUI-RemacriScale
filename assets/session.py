import os
import time
from typing import Optional

import onnxruntime as ort
import torch


class SessionMixin:
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
