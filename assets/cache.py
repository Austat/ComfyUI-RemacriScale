import json
import os
import shutil
import onnxruntime as ort
import torch


class RuntimeCacheMixin:
    @classmethod
    def _clear_sessions(cls):
        with cls._session_lock:
            cls._sessions.clear()
            cls._active_providers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @classmethod
    def _check_runtime_versions_and_invalidate_cache(cls):
        current = {
            "torch": torch.__version__, "cuda": torch.version.cuda,
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
                pass
        if previous == current:
            return
        print("[RemacriOnnxUpscale] Runtime versions changed; invalidating TensorRT caches.")
        for cache_path in (cls.TRT_TIMING_CACHE_PATH, cls.TRT_ENGINE_CACHE_PATH):
            if os.path.isdir(cache_path):
                try:
                    shutil.rmtree(cache_path)
                except OSError as exc:
                    print(f"[RemacriOnnxUpscale] Failed to delete {cache_path}: {exc}")
        try:
            with open(cls.RUNTIME_VERSION_FILE, "w", encoding="utf-8") as handle:
                json.dump(current, handle, indent=2)
        except OSError as exc:
            print(f"[RemacriOnnxUpscale] Failed to write runtime metadata: {exc}")
        cls._clear_sessions()
        cls._metadata_cache.clear()
