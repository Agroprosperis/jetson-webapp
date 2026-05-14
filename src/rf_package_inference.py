import json
import os

import cv2
import numpy as np
import tensorrt as trt
import torch


def _trt_to_torch_dtype(trt_dtype):
    mapping = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
    }
    assert trt_dtype in mapping, f"Unsupported TensorRT tensor dtype: {trt_dtype}"
    return mapping[trt_dtype]


class RoboflowPackageDetector:
    BOXES_OUTPUT_NAME = "dets"
    SCORES_OUTPUT_NAME = "labels"

    def __init__(self, engine_path):
        self.engine_path = engine_path
        _, extension = os.path.splitext(engine_path)
        assert extension.lower() == ".engine", f"RF package model must be a TensorRT engine: {engine_path}"
        self.config_path = f"{engine_path}.json"

        with open(self.config_path, "r", encoding="utf-8") as config_input:
            self.config = json.load(config_input)

        self.network_input = self.config["network_input"]
        assert isinstance(self.network_input, dict)

        training_input_size = self.network_input["training_input_size"]
        assert isinstance(training_input_size, dict)
        self.input_width = int(training_input_size["width"])
        self.input_height = int(training_input_size["height"])
        assert self.input_width > 0 and self.input_height > 0

        self.resize_mode = self.network_input["resize_mode"]
        assert self.resize_mode == "stretch", f"Unsupported RF package resize_mode: {self.resize_mode}"

        self.color_mode = self.network_input["color_mode"]
        assert self.color_mode in ("rgb", "bgr"), f"Unsupported RF package color_mode: {self.color_mode}"

        self.input_channels = int(self.network_input["input_channels"])
        assert self.input_channels == 3

        self.scaling_factor = float(self.network_input["scaling_factor"])
        self.normalization = self.network_input["normalization"]
        assert isinstance(self.normalization, list) and len(self.normalization) >= 2

        self.class_names_path = f"{engine_path}.class_names.txt"
        self.class_names = self._load_class_names(self.class_names_path)

        self.logger_trt = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger_trt)
        with open(engine_path, "rb") as engine_input:
            self.engine = self.runtime.deserialize_cuda_engine(engine_input.read())
        assert self.engine is not None, f"Failed to deserialize TensorRT engine: {engine_path}"

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = self._allocate_bindings()
        assert len(self.inputs) == 1, f"RF package engine must have exactly one input, got {len(self.inputs)}"
        assert len(self.outputs) >= 2, f"RF package engine must expose boxes and scores outputs, got {len(self.outputs)}"
        self.boxes_output, self.scores_output = self._resolve_output_tensors()
        self._validate_detection_output_shapes()

    def _load_class_names(self, class_names_path):
        with open(class_names_path, "r", encoding="utf-8") as class_names_input:
            return [line.strip() for line in class_names_input if line.strip()]

    def _allocate_bindings(self):
        inputs = []
        outputs = []
        bindings = [None] * self.engine.num_io_tensors

        for tensor_index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(tensor_index)
            is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
            shape = tuple(self.engine.get_tensor_shape(name))

            if is_input and -1 in shape:
                shape = (1, self.input_channels, self.input_height, self.input_width)
                self.context.set_input_shape(name, shape)
            elif -1 in shape:
                shape = tuple(self.context.get_tensor_shape(name))
            assert -1 not in shape, f"Failed to resolve dynamic TensorRT tensor shape for {name}: {shape}"

            dtype = _trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            tensor = torch.empty(shape, dtype=dtype, device="cuda").contiguous()
            binding = {
                "name": name,
                "tensor": tensor,
                "ptr": tensor.data_ptr(),
                "shape": shape,
                "is_input": is_input,
            }
            bindings[tensor_index] = binding["ptr"]
            if is_input:
                inputs.append(binding)
            else:
                outputs.append(binding)

        return inputs, outputs, bindings

    def _resolve_output_tensors(self):
        outputs_by_name = {output["name"]: output["tensor"] for output in self.outputs}
        missing_names = [
            name
            for name in (self.BOXES_OUTPUT_NAME, self.SCORES_OUTPUT_NAME)
            if name not in outputs_by_name
        ]
        assert not missing_names, (
            "RF package engine is missing required detection outputs "
            f"{missing_names}; available outputs: {sorted(outputs_by_name)}"
        )
        return outputs_by_name[self.BOXES_OUTPUT_NAME], outputs_by_name[self.SCORES_OUTPUT_NAME]

    def _validate_detection_output_shapes(self):
        assert self.boxes_output.ndim == 3, (
            f"RF package boxes output '{self.BOXES_OUTPUT_NAME}' must have shape [1, N, 4], "
            f"got {tuple(self.boxes_output.shape)}"
        )
        assert self.scores_output.ndim == 3, (
            f"RF package scores output '{self.SCORES_OUTPUT_NAME}' must have shape [1, N, C], "
            f"got {tuple(self.scores_output.shape)}"
        )
        assert self.boxes_output.shape[0] == 1 and self.scores_output.shape[0] == 1, (
            "RF package detection outputs must use batch size 1, "
            f"got boxes={tuple(self.boxes_output.shape)} scores={tuple(self.scores_output.shape)}"
        )
        assert self.boxes_output.shape[2] == 4, (
            f"RF package boxes output '{self.BOXES_OUTPUT_NAME}' must end with 4 box values, "
            f"got {tuple(self.boxes_output.shape)}"
        )
        assert self.scores_output.shape[1] == self.boxes_output.shape[1], (
            "RF package boxes and scores outputs must have the same number of predictions, "
            f"got boxes={tuple(self.boxes_output.shape)} scores={tuple(self.scores_output.shape)}"
        )
        assert self.scores_output.shape[2] > 0, (
            f"RF package scores output '{self.SCORES_OUTPUT_NAME}' must contain class logits, "
            f"got {tuple(self.scores_output.shape)}"
        )

    def _preprocess(self, frame_bgr):
        image = cv2.resize(frame_bgr, (self.input_width, self.input_height))
        if self.color_mode == "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / self.scaling_factor

        mean = np.asarray(self.normalization[0], dtype=np.float32).reshape(1, 1, -1)
        std = np.asarray(self.normalization[1], dtype=np.float32).reshape(1, 1, -1)
        image = (image - mean) / std

        image = np.transpose(image, (2, 0, 1))[None, ...]
        input_tensor = torch.from_numpy(np.ascontiguousarray(image)).to(device="cuda")
        return input_tensor.to(dtype=self.inputs[0]["tensor"].dtype)

    def _postprocess(self, orig_width, orig_height, conf_thresh):
        boxes = self.boxes_output
        scores = self.scores_output

        probabilities = torch.sigmoid(scores.float().clamp(-100, 100))
        confidences, class_ids = torch.max(probabilities[0], dim=-1)
        keep = confidences > float(conf_thresh)
        if not torch.any(keep):
            return np.zeros((0, 6), dtype=np.float32)

        kept_boxes = boxes[0][keep].float()
        kept_confidences = confidences[keep]
        kept_class_ids = class_ids[keep].float()

        cx, cy, width, height = kept_boxes.unbind(-1)
        x1 = (cx - 0.5 * width) * float(orig_width)
        y1 = (cy - 0.5 * height) * float(orig_height)
        x2 = (cx + 0.5 * width) * float(orig_width)
        y2 = (cy + 0.5 * height) * float(orig_height)

        xyxy = torch.stack((x1, y1, x2, y2), dim=1)
        xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clamp(0, float(orig_width))
        xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clamp(0, float(orig_height))
        detections = torch.cat(
            (
                xyxy,
                kept_confidences.reshape(-1, 1),
                kept_class_ids.reshape(-1, 1),
            ),
            dim=1,
        )
        return detections.detach().cpu().numpy().astype(np.float32, copy=False)

    def predict_boxes(self, frame_bgr, conf_thresh):
        orig_height, orig_width = frame_bgr.shape[:2]
        input_tensor = self._preprocess(frame_bgr)
        self.inputs[0]["tensor"].copy_(input_tensor)
        self.context.execute_v2(self.bindings)
        return self._postprocess(orig_width, orig_height, conf_thresh)
