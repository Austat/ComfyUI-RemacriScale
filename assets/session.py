import os
import time
import onnxruntime as ort


class SessionMixin:
    @classmethod
    def _try_create_session(cls, model_path, provider, input_name, batch_size,
                            channels, height, width, timing_cache_path,
                            diagnostics, device_id):
        if provider not in ort.get_available_providers():
            print(f"[RemacriOnnxUpscale] Provider unavailable: {provider}")
            return None
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, min(os.cpu_count() or 8, 16))
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if diagnostics:
            opts.enable_profiling = True
            opts.log_severity_level = 1
        did = 0 if device_id is None else int(device_id)
        cuda_opts = {"device_id": did, "arena_extend_strategy": "kSameAsRequested",
                     "cudnn_conv_algo_search": "EXHAUSTIVE",
                     "do_copy_in_default_stream": True}
        if provider == "MIGraphXExecutionProvider":
            providers = [(provider, {"device_id": str(did)}), "CPUExecutionProvider"]
        elif provider == "ROCmExecutionProvider":
            providers = [(provider, {"device_id": str(did),
                                     "arena_extend_strategy": "kSameAsRequested",
                                     "do_copy_in_default_stream": "1"}),
                         "CPUExecutionProvider"]
        elif provider == "TensorrtExecutionProvider":
            engine = os.path.join(cls.TRT_ENGINE_CACHE_PATH, f"device_{did}")
            timing = os.path.join(os.path.dirname(timing_cache_path), f"device_{did}",
                                  os.path.basename(timing_cache_path))
            os.makedirs(engine, exist_ok=True)
            os.makedirs(os.path.dirname(timing), exist_ok=True)
            minimum = f"{input_name}:1x{channels}x{height}x{width}"
            optimum = f"{input_name}:{batch_size}x{channels}x{height}x{width}"
            trt = {"device_id": did, "trt_engine_cache_enable": True,
                   "trt_engine_cache_path": engine, "trt_timing_cache_enable": True,
                   "trt_timing_cache_path": timing, "trt_fp16_enable": True,
                   "trt_int8_enable": False, "trt_dla_enable": False,
                   "trt_max_workspace_size": 8 * 1024**3,
                   "trt_profile_min_shapes": minimum,
                   "trt_profile_opt_shapes": optimum,
                   "trt_profile_max_shapes": optimum}
            providers = [(provider, trt), ("CUDAExecutionProvider", cuda_opts),
                         "CPUExecutionProvider"]
        elif provider == "CUDAExecutionProvider":
            providers = [(provider, cuda_opts), "CPUExecutionProvider"]
        else:
            providers = [provider]
        started = time.perf_counter()
        try:
            session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
            label = "CPU" if device_id is None else f"GPU {device_id}"
            print(f"[RemacriOnnxUpscale] Initialized {label}: {session.get_providers()} "
                  f"in {time.perf_counter()-started:.2f}s")
            return session
        except Exception as exc:
            print(f"[RemacriOnnxUpscale] GPU {did} provider {provider} failed: {exc}")
            return None

    @classmethod
    def _load_session(cls, model_path, provider, input_name, batch_size, channels,
                      height, width, timing_cache_path, diagnostics, device_id):
        key = (os.path.abspath(model_path), provider, device_id, input_name,
               batch_size, channels, height, width, bool(diagnostics))
        with cls._session_lock:
            if key in cls._sessions:
                return cls._sessions[key], cls._active_providers[key]
        chain = [provider]
        if provider == "MIGraphXExecutionProvider":
            chain += ["ROCmExecutionProvider", "CPUExecutionProvider"]
        elif device_id is not None:
            chain += ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            chain += ["CPUExecutionProvider"]
        for candidate in dict.fromkeys(chain):
            session = cls._try_create_session(model_path, candidate, input_name,
                batch_size, channels, height, width, timing_cache_path,
                diagnostics, device_id)
            if session is not None:
                active = session.get_providers()[0]
                with cls._session_lock:
                    cls._sessions[key] = session
                    cls._active_providers[key] = active
                return session, active
        raise RuntimeError(f"All ONNX Runtime providers failed for device {device_id}.")
