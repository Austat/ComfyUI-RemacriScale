import torch
import cv2
import numpy as np
import os
import warnings
from tqdm import tqdm

# Import methods
from .downsample_methods.frame_drop import frame_drop
from .downsample_methods.frame_blend import frame_blend
from .downsample_methods.optical_flow import optical_flow

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


class DownsampleFPSNode:
    @classmethod
    def INPUT_TYPES(cls):
        max_cores = os.cpu_count()

        return {
            "required": {
                "frames": ("IMAGE",),
                "input_fps": ("INT", {"default": 48, "min": 1, "max": 240}),
                "target_fps": ("INT", {"default": 24, "min": 1, "max": 240}),
                "method": ([
                    "Frame dropping",
                    "Frame blending (CPU)",
                    "Optical flow (CPU)",
                ],),
                "cpu_threads": (
                    ["auto"] + [str(i) for i in range(1, max_cores + 1)],
                    {"default": "auto"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("downsampled_frames",)
    FUNCTION = "downsample"
    CATEGORY = "Video"

    def __init__(self):
        print("\n=== DownsampleFPSNode STATUS ===")
        print("Using CPU-only methods (no OpenCV CUDA).")
        print(f"CPU cores detected: {os.cpu_count()}")
        print("Methods: Frame dropping, CPU blending, CPU optical flow.\n")

    def downsample(self, frames, input_fps, target_fps, method, cpu_threads):

        if target_fps >= input_fps:
            print("Target FPS is equal or higher than input FPS. No downsampling applied.")
            return (frames,)

        np_frames = (frames.cpu().numpy() * 255).astype(np.uint8)

        ratio = input_fps / target_fps
        frame_count = int(len(np_frames) / ratio)
        indices = [int(i * ratio) for i in range(frame_count)]

        print(f"[DownsampleFPSNode] Method: {method}, "
              f"input_fps={input_fps}, target_fps={target_fps}, "
              f"in_frames={len(np_frames)}, out_frames={len(indices)}")

        threads = os.cpu_count() if cpu_threads == "auto" else int(cpu_threads)

        # Dispatch to method modules
        if method == "Frame dropping":
            selected = frame_drop(np_frames, indices)

        elif method == "Frame blending (CPU)":
            selected = frame_blend(np_frames, indices, threads)

        elif method == "Optical flow (CPU)":
            selected = optical_flow(np_frames, indices, threads)

        selected = np.stack(selected).astype(np.float32) / 255.0
        out_tensor = torch.from_numpy(selected)

        return (out_tensor,)
