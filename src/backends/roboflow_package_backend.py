import time

import numpy as np
import torch
from ultralytics.engine.results import Boxes

from inference_pipeline import BotSortTrackerBackend, LOGGER, MockResult
from rf_package_inference import RoboflowPackageDetector


class RoboflowPackageBackend(BotSortTrackerBackend):
    def __init__(self, model_path: str, args):
        super().__init__()
        LOGGER.info("Initializing RF package TensorRT backend with model: %s", model_path)
        self.detector = RoboflowPackageDetector(model_path)
        args.class_names = self.detector.class_names
        LOGGER.info(
            "RF package class names loaded from %s: %s",
            self.detector.class_names_path,
            self.detector.class_names,
        )
        args.model_inference_width = self.detector.input_width
        args.model_inference_height = self.detector.input_height
        LOGGER.info(
            "RF package runtime settings: size=%dx%d resize=%s color=%s",
            self.detector.input_width,
            self.detector.input_height,
            self.detector.resize_mode,
            self.detector.color_mode,
        )

    def _detect_objects(self, frame: np.ndarray, args):
        model_conf = getattr(args, "model_conf", 0.5)
        start_time = time.perf_counter()
        predictions = self.detector.predict_boxes(frame, model_conf)
        end_time = time.perf_counter()

        h, w = frame.shape[:2]
        if predictions.size == 0:
            empty_boxes = Boxes(torch.zeros((0, 6)), orig_shape=(h, w))
            return [MockResult(empty_boxes)], end_time - start_time

        boxes_obj = Boxes(torch.from_numpy(predictions), orig_shape=(h, w))
        return [MockResult(boxes_obj)], end_time - start_time
