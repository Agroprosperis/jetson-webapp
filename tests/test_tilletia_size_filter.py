import argparse
import sys
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference_pipeline import (
    BotSortTrackerBackend,
    _filter_oversized_tilletia_detections,
)


class FakeBoxes:
    def __init__(self, rows):
        self.data = np.asarray(rows, dtype=np.float32).reshape(-1, 6)

    def __len__(self):
        return len(self.data)

    @property
    def xyxy(self):
        return self.data[:, :4]

    @property
    def cls(self):
        return self.data[:, 5]

    def __getitem__(self, index):
        return FakeBoxes(self.data[index])

    def cpu(self):
        return self

    def numpy(self):
        return self


class FakeResult:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeTracker:
    def __init__(self):
        self.received_boxes = None

    def update(self, boxes, frame):
        self.received_boxes = boxes
        return boxes.data


class TilletiaSizeFilterTests(unittest.TestCase):
    def test_filters_tilletia_when_either_side_exceeds_training_limit(self):
        boxes = FakeBoxes(
            [
                [0, 0, 68, 68, 0.9, 1],
                [0, 0, 69, 20, 0.9, 1],
                [0, 0, 20, 69, 0.9, 1],
            ]
        )
        args = argparse.Namespace(class_names=["Alternaria", "Tilletia"])

        filtered = _filter_oversized_tilletia_detections(boxes, (1944, 2592, 3), args)

        self.assertEqual(len(filtered), 1)
        np.testing.assert_array_equal(filtered.data[0, :4], [0, 0, 68, 68])

    def test_rescales_width_and_height_limits_independently(self):
        boxes = FakeBoxes(
            [
                [0, 0, 34, 17, 0.9, 1],
                [0, 0, 35, 17, 0.9, 1],
                [0, 0, 34, 18, 0.9, 1],
            ]
        )
        args = argparse.Namespace(class_names={1: "Tilletia"})

        filtered = _filter_oversized_tilletia_detections(boxes, (486, 1296, 3), args)

        self.assertEqual(len(filtered), 1)
        np.testing.assert_array_equal(filtered.data[0, :4], [0, 0, 34, 17])

    def test_keeps_oversized_non_tilletia_detection(self):
        boxes = FakeBoxes([[0, 0, 100, 100, 0.9, 0]])
        args = argparse.Namespace(class_names=["Alternaria", "Tilletia"])

        filtered = _filter_oversized_tilletia_detections(boxes, (1944, 2592, 3), args)

        self.assertEqual(len(filtered), 1)

    def test_accepts_empty_detections(self):
        boxes = FakeBoxes([])
        args = argparse.Namespace(class_names=["Tilletia"])

        filtered = _filter_oversized_tilletia_detections(boxes, (1944, 2592, 3), args)

        self.assertIs(filtered, boxes)

    def test_shared_tracker_path_receives_only_filtered_detections(self):
        backend = BotSortTrackerBackend()
        backend.tracker = FakeTracker()
        boxes = FakeBoxes(
            [
                [0, 0, 68, 68, 0.9, 1],
                [0, 0, 69, 68, 0.9, 1],
            ]
        )
        args = argparse.Namespace(class_names=["Alternaria", "Tilletia"])
        frame = np.zeros((1944, 2592, 3), dtype=np.uint8)

        tracks, _ = backend._track([FakeResult(boxes)], frame, args)

        self.assertEqual(len(backend.tracker.received_boxes), 1)
        np.testing.assert_array_equal(tracks[0, :4], [0, 0, 68, 68])

