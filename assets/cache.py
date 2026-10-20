import json
import os
import shutil

import onnxruntime as ort
import torch


class RuntimeCacheMixin:
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
