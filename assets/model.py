import os
import onnxruntime as ort
import folder_paths


class ModelMixin:
    @staticmethod
    def _find_model(model_file):
        for directory in folder_paths.get_folder_paths("upscale_models"):
            model_path = os.path.join(directory, model_file)
            if os.path.isfile(model_path):
                return model_path
        raise FileNotFoundError(f"Model '{model_file}' not found.")

    @classmethod
    def _model_metadata(cls, model_path):
        cache_key = os.path.abspath(model_path)
        cached = cls._metadata_cache.get(cache_key)
        if cached is not None:
            return cached
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        metadata_session = ort.InferenceSession(
            model_path, sess_options=options, providers=["CPUExecutionProvider"]
        )
        model_input = metadata_session.get_inputs()[0]
        model_output = metadata_session.get_outputs()[0]
        metadata = (
            model_input.name, tuple(model_input.shape),
            model_output.name, tuple(model_output.shape),
        )
        cls._metadata_cache[cache_key] = metadata
        return metadata

    @classmethod
    def _validate_model_input(cls, model_path, requested_batch_size, channels, height, width):
        input_name, input_shape, output_name, output_shape = cls._model_metadata(model_path)
        if len(input_shape) != 4:
            raise RuntimeError(f"Expected a four-dimensional NCHW model input, got {input_shape!r}.")
        fixed_batch = input_shape[0] if isinstance(input_shape[0], int) else None
        if fixed_batch is not None and fixed_batch != requested_batch_size:
            raise RuntimeError(
                f"The ONNX model has fixed batch={fixed_batch}, but batch_size={requested_batch_size}. "
                "Export the model with a dynamic batch dimension for true batching."
            )
        expected = (channels, height, width)
        axis_names = {1: "channels", 2: "height", 3: "width"}
        for axis, (dimension, actual) in enumerate(zip(input_shape[1:], expected), start=1):
            if isinstance(dimension, int) and dimension != actual:
                raise RuntimeError(f"The model requires {axis_names[axis]}={dimension}, got {actual}.")
        return input_name, input_shape, output_name, output_shape
