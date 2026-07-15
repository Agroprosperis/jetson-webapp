import argparse
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import visualize


class CaptionModeTests(unittest.TestCase):
    def setUp(self):
        visualize.reset_object_counter()
        self.frame = np.zeros((32, 32, 3), dtype=np.uint8)
        self.tracks = np.asarray(
            [[2, 3, 20, 21, 1, 0.9, 0, 0]],
            dtype=np.float32,
        )

    def render(self, args):
        with (
            patch.object(visualize, "BOX_ANNOTATOR") as box_annotator,
            patch.object(visualize, "LABEL_ANNOTATOR") as label_annotator,
            patch.object(visualize, "draw_combined_banner") as draw_banner,
        ):
            box_annotator.annotate.return_value = self.frame
            label_annotator.annotate.return_value = self.frame
            draw_banner.return_value = self.frame

            visualize.visualize_frame_with_supervision(
                self.frame,
                self.tracks,
                args,
            )

            return box_annotator, label_annotator, draw_banner

    def test_captions_are_disabled_when_argument_is_omitted(self):
        args = argparse.Namespace(
            class_names=["Tilletia"],
            pipeline_id="test",
            vis_strategy="tracker",
        )

        box_annotator, label_annotator, draw_banner = self.render(args)

        box_annotator.annotate.assert_called_once()
        label_annotator.annotate.assert_not_called()
        draw_banner.assert_called_once()

    def test_captions_can_be_enabled(self):
        args = argparse.Namespace(
            captions_enabled=True,
            class_names=["Tilletia"],
            pipeline_id="test",
            vis_strategy="tracker",
        )

        box_annotator, label_annotator, draw_banner = self.render(args)

        box_annotator.annotate.assert_called_once()
        label_annotator.annotate.assert_called_once()
        labels = label_annotator.annotate.call_args.kwargs["labels"]
        self.assertEqual(labels, ["Tilletia 0.90 id:1"])
        draw_banner.assert_called_once()


if __name__ == "__main__":
    unittest.main()
