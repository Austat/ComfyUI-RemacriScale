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
    Custom ComfyUI node for image upscaling using ONNX models.
    """

    _session = None
    _model_path = None
    _provider = None

    @classmethod
    def INPUT_TYPES(cls):

        # Collect all .onnx files from ALL upscale_models paths
        files = []
        for d in folder_paths.get_folder_paths("upscale_models"):
            if os.path.isdir(d):
                for f in os.listdir(d):
                    if f.lower().endswith(".onnx") and f not in files:
                        files.append(f)

        if not files:
            files = ["(no .onnx models found)"]

        providers = [
            "TensorrtExecutionProvider",
            "CUDAExecutionProvider",
            "CPUExecutionProvider"
        ]

        return {
            "required": {
                "image": ("IMAGE",),
                "model_file": (files,),
                "provider": (providers,),
                "final_resolution": (["hd", "fhd", "no downscaling"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("upsampled",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscale"
    OUTPUT_NODE = True

    @classmethod
    def _load_session(cls, model_path, provider):

        if cls._session is None or cls._model_path != model_path or cls._provider != provider:
            cls._session = ort.InferenceSession(model_path, providers=[provider])
            cls._model_path = model_path
            cls._provider = provider
            print(f"[RemacriOnnxUpscale] Using provider: {provider}")

        return cls._session

    def upscale(self, image, model_file, provider, final_resolution, progress=None):

        # Find model in ANY upscale_models folder
        model_path = None
        search_dirs = folder_paths.get_folder_paths("upscale_models")

        for d in search_dirs:
            p = os.path.join(d, model_file)
            if os.path.exists(p):
                model_path = p
                break

        if model_path is None:
            raise FileNotFoundError(
                f"Model '{model_file}' not found in any upscale_models directory:\n" +
                "\n".join(search_dirs)
            )

        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)

        session = self._load_session(model_path, provider)

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

        for i in range(total):

            arr = (image[i].cpu().numpy() * 255).astype(np.uint8)
            inp = arr.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

            ort_inputs = {session.get_inputs()[0].name: inp}
            ort_outs = session.run(None, ort_inputs)

            out = ort_outs[0][0].transpose(1, 2, 0)

            if final_resolution == "hd":
                out = cv2.resize(out, (1280, 720), interpolation=cv2.INTER_AREA)
            elif final_resolution == "fhd":
                out = cv2.resize(out, (1920, 1080), interpolation=cv2.INTER_AREA)

            out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
            out = np.clip(out, 0.0, 1.0)

            out_batch.append(out)

            percent = int(((i + 1) / total) * 100)

            if progress is not None:
                progress(percent)

            pbar.update(percent - pbar.n)
            pbar.set_description(f"Upscaling (Image {i+1}/{total})")

        pbar.close()

        out = np.stack(out_batch, axis=0).astype(np.float32)
        out_tensor = torch.from_numpy(out).float()

        return (out_tensor,)
