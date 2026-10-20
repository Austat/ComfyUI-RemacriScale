import numpy as np
import torch


class InferenceMixin:
    @staticmethod
    def _target_size(final_resolution):
            # torch interpolation uses (height, width)
            return {
                "hd": (720, 1280),
                "fhd": (1080, 1920),
                "2k": (1440, 2560),
                "4k": (2160, 3840),
                "8k": (4320, 7680),
                # Common square (1:1) output presets.
                "512x512": (512, 512),
                "768x768": (768, 768),
                "1024x1024": (1024, 1024),
                "2048x2048": (2048, 2048),
                "4096x4096": (4096, 4096),
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
