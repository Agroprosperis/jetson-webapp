from __future__ import annotations

import logging
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LOGGER = logging.getLogger("grid_runtime")

DEFAULT_GRID_COLOR = (0, 255, 255)
DEFAULT_LINEA_CONFIG = "configs/linea/linea_hgnetv2_s.py"
DEFAULT_LINEA_CHECKPOINT = "weights/linea_hgnetv2_s.pth"
DEFAULT_APP_CHECKPOINT = "/app/model/linea/linea_hgnetv2_s.pth"


@dataclass(slots=True)
class GridOverlay:
    analysis_shape: tuple[int, int]
    lines: dict[str, list[dict[str, Any]]]
    viewport_corners: list[tuple[int, int]] | None
    score: float | None
    score_raw: float | None


class GridProcessor:
    def __init__(
        self,
        color: tuple[int, int, int] = DEFAULT_GRID_COLOR,
        detect_every_n_frames: int = 1,
        score_ema_alpha: float = 0.25,
    ) -> None:
        self._color = color
        self._detect_every_n_frames = max(int(detect_every_n_frames), 1)
        self._score_ema_alpha = min(max(float(score_ema_alpha), 0.0), 1.0)
        self._detection_warmup_frames = 3
        self._processed_frames = 0
        self._last_gap_info: dict[str, list[float]] = {
            "vertical": [],
            "horizontal": [],
        }
        self._last_score: float | None = None
        self._last_score_raw: float | None = None
        self._initialized = False
        self._init_failed = False
        self._processing_failed = False
        self._preprocess = None
        self._detector = None
        self._clusterizer = None
        self._regularizer = None
        self._gap_analyzer = None
        self._tracker_cls = None
        self._accumulator_cls = None
        self._grid_builder_cls = None
        self._tracker = None
        self._accumulator = None
        self._grid_builder = None

    def process(self, frame_bgr: np.ndarray) -> GridOverlay | None:
        if self._processing_failed:
            return None
        if not self._ensure_initialized():
            return None

        try:
            analysis_rgb = self._preprocess(frame_bgr)
            analysis_shape = analysis_rgb.shape[:2]
            analysis_gray = cv2.cvtColor(analysis_rgb, cv2.COLOR_RGB2GRAY)
            flow_shift = self._tracker.estimate_flow_shift(analysis_gray)
            self._processed_frames += 1
            run_full_detection = self._should_run_full_detection()
            score_raw: float | None = None

            if run_full_detection:
                raw_lines = self._detector(analysis_rgb)
                accumulated_lines = self._accumulator(raw_lines, flow_shift)
                clustered_lines = self._clusterizer(accumulated_lines, analysis_shape)
                regularized_lines = self._regularizer(clustered_lines)
                gap_info = self._gap_analyzer(regularized_lines)
                current_grid = self._grid_builder.build_current(regularized_lines, gap_info)
                score_raw = self._compute_grid_score(current_grid)
                self._last_score_raw = score_raw
                self._last_score = self._update_score_ema(score_raw)
                self._last_gap_info = {
                    "vertical": list(gap_info["vertical"]),
                    "horizontal": list(gap_info["horizontal"]),
                }
            else:
                gap_info = {
                    "vertical": list(self._last_gap_info["vertical"]),
                    "horizontal": list(self._last_gap_info["horizontal"]),
                }
                self._accumulator([], flow_shift)
                current_grid = self._empty_grid_state()

            tracker_input = current_grid["accepted"]["vertical"] + current_grid["accepted"]["horizontal"]
            tracker_state = self._tracker(tracker_input, flow_shift=flow_shift)

            if run_full_detection:
                grid_state = self._grid_builder.integrate_predictions(current_grid, gap_info, tracker_state)
                overlay_lines = self._overlay_lines_from_grid_state(grid_state)
            else:
                overlay_lines = self._tracking_only_overlay_lines(current_grid, gap_info, tracker_state)
            overlay_lines = self._filter_lines_within_frame(overlay_lines, analysis_shape)

            viewport_corners = self._build_viewport_corners(
                overlay_lines,
                analysis_shape,
                frame_bgr.shape[:2],
            )
            return GridOverlay(
                analysis_shape=analysis_shape,
                lines=overlay_lines,
                viewport_corners=viewport_corners,
                score=self._last_score,
                score_raw=score_raw,
            )
        except Exception:
            self._processing_failed = True
            LOGGER.exception("Grid processing failed. Disabling grid detection for this run.")
            return None

    def render(self, frame_bgr: np.ndarray, overlay: GridOverlay | None) -> np.ndarray:
        if overlay is None:
            return frame_bgr

        for orientation in ("vertical", "horizontal"):
            for line_state in overlay.lines.get(orientation, []):
                self._draw_grid_line(frame_bgr, overlay.analysis_shape, orientation, line_state)

        corners = overlay.viewport_corners
        if corners and len(corners) == 4:
            for idx in range(4):
                cv2.line(
                    frame_bgr,
                    corners[idx],
                    corners[(idx + 1) % 4],
                    self._color,
                    2,
                    cv2.LINE_8,
                )

        return frame_bgr

    def reset_state(self, *, preserve_score: bool = True) -> None:
        self._processed_frames = 0
        self._last_gap_info = {
            "vertical": [],
            "horizontal": [],
        }
        if not preserve_score:
            self._last_score = None
        self._last_score_raw = None
        if self._tracker_cls is not None:
            self._tracker = self._tracker_cls()
            self._detection_warmup_frames = max(int(self._tracker.warmup_frames), 0)
        if self._accumulator_cls is not None:
            self._accumulator = self._accumulator_cls()
        if self._grid_builder_cls is not None:
            self._grid_builder = self._grid_builder_cls()

    def _ensure_initialized(self) -> bool:
        if self._initialized:
            return True
        if self._init_failed:
            return False

        try:
            self._append_linea_import_path()
            from grid import (
                FramePreprocessor,
                GapAnalyzer,
                GridBuilder,
                GridTracker,
                LineClusterizer,
                LineDetector,
                LineRegularizer,
                TemporalLineAccumulator,
            )

            config_path = self._resolve_linea_config_path()
            checkpoint_path = self._resolve_linea_checkpoint_path()

            self._preprocess = FramePreprocessor()
            self._detector = LineDetector(
                config_path=config_path,
                checkpoint_path=checkpoint_path,
            )
            self._clusterizer = LineClusterizer()
            self._regularizer = LineRegularizer()
            self._gap_analyzer = GapAnalyzer()
            self._tracker_cls = GridTracker
            self._accumulator_cls = TemporalLineAccumulator
            self._grid_builder_cls = GridBuilder
            self.reset_state(preserve_score=False)
            self._initialized = True
            LOGGER.info(
                "Grid processor initialized with config=%s checkpoint=%s",
                config_path,
                checkpoint_path,
            )
            return True
        except Exception:
            self._init_failed = True
            LOGGER.exception("Grid processor initialization failed.")
            return False

    def _should_run_full_detection(self) -> bool:
        if self._detect_every_n_frames <= 1:
            return True
        if self._processed_frames <= self._detection_warmup_frames:
            return True
        post_warmup_frame = self._processed_frames - self._detection_warmup_frames
        return post_warmup_frame % self._detect_every_n_frames == 0

    def _empty_grid_state(self) -> dict[str, dict[str, list[dict[str, Any]]]]:
        return {
            "accepted": {"vertical": [], "horizontal": []},
            "rejected": {"vertical": [], "horizontal": []},
        }

    def _compute_grid_score(
        self,
        current_grid: dict[str, dict[str, list[dict[str, Any]]]],
    ) -> float | None:
        orientation_scores: list[float] = []
        any_evidence = False

        for orientation in ("vertical", "horizontal"):
            accepted = current_grid["accepted"][orientation]
            rejected = current_grid["rejected"][orientation]
            total_count = len(accepted) + len(rejected)
            accepted_support = self._line_support_sum(accepted)
            rejected_support = self._line_support_sum(rejected)
            total_support = accepted_support + rejected_support
            if total_count > 0 or total_support > 1e-9:
                any_evidence = True

            count_ratio = 1.0 if total_count <= 0 else float(len(accepted)) / float(total_count)
            support_ratio = 1.0 if total_support <= 1e-9 else accepted_support / total_support
            # Weight support more than count so sparse-but-clean grids still score well.
            orientation_scores.append(0.25 * count_ratio + 0.75 * support_ratio)

        if not any_evidence:
            return None
        return math.sqrt(max(orientation_scores[0], 0.0) * max(orientation_scores[1], 0.0))

    def _update_score_ema(self, score: float | None) -> float | None:
        if score is None:
            return self._last_score
        if self._last_score is None:
            return score
        alpha = self._score_ema_alpha
        return ((1.0 - alpha) * float(self._last_score)) + (alpha * float(score))

    def _line_support_sum(self, lines: list[dict[str, Any]]) -> float:
        return sum(max(float(line.get("conf", 0.0)), 0.0) for line in lines)

    def _overlay_lines_from_grid_state(
        self,
        grid_state: dict[str, dict[str, list[dict[str, Any]]]],
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "vertical": list(grid_state["accepted"]["vertical"]),
            "horizontal": list(grid_state["accepted"]["horizontal"]),
        }

    def _tracking_only_overlay_lines(
        self,
        current_grid: dict[str, dict[str, list[dict[str, Any]]]],
        gap_info: dict[str, list[float]],
        tracker_state: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        reference_lines = {
            "vertical": sorted(
                [dict(line) for line in tracker_state["reference"]["vertical"]],
                key=lambda item: item["rho"],
            ),
            "horizontal": sorted(
                [dict(line) for line in tracker_state["reference"]["horizontal"]],
                key=lambda item: item["rho"],
            ),
        }
        if reference_lines["vertical"] or reference_lines["horizontal"]:
            return reference_lines

        grid_state = self._grid_builder.integrate_predictions(current_grid, gap_info, tracker_state)
        return self._overlay_lines_from_grid_state(grid_state)

    def _filter_lines_within_frame(
        self,
        lines: dict[str, list[dict[str, Any]]],
        analysis_shape: tuple[int, int],
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "vertical": [
                dict(line)
                for line in lines.get("vertical", [])
                if self._line_rho_in_bounds("vertical", line, analysis_shape)
            ],
            "horizontal": [
                dict(line)
                for line in lines.get("horizontal", [])
                if self._line_rho_in_bounds("horizontal", line, analysis_shape)
            ],
        }

    def _line_rho_in_bounds(
        self,
        orientation: str,
        line_state: dict[str, Any],
        analysis_shape: tuple[int, int],
    ) -> bool:
        analysis_h, analysis_w = analysis_shape
        rho = float(line_state["rho"])
        if orientation == "vertical":
            return 0.0 <= rho <= float(max(analysis_w - 1, 0))
        return 0.0 <= rho <= float(max(analysis_h - 1, 0))

    def _append_linea_import_path(self) -> None:
        for candidate in self._linea_root_candidates():
            path_str = str(candidate)
            if candidate.exists() and path_str not in sys.path:
                sys.path.append(path_str)

    def _resolve_linea_config_path(self) -> str:
        env_path = self._existing_path_from_env("GRID_LINEA_CONFIG")
        if env_path is not None:
            return env_path

        linea_root = self._resolve_linea_root()
        if linea_root is None:
            raise FileNotFoundError("LINEA runtime files are not available.")

        config_path = linea_root / DEFAULT_LINEA_CONFIG
        if config_path.exists():
            return str(config_path)
        raise FileNotFoundError(f"Missing LINEA config: {config_path}")

    def _resolve_linea_checkpoint_path(self) -> str:
        env_path = self._existing_path_from_env("GRID_LINEA_CHECKPOINT")
        if env_path is not None:
            return env_path

        candidates = [Path(DEFAULT_APP_CHECKPOINT)]
        linea_root = self._resolve_linea_root()
        if linea_root is not None:
            candidates.append(linea_root / DEFAULT_LINEA_CHECKPOINT)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            "Grid checkpoint not found. Expected "
            f"{DEFAULT_APP_CHECKPOINT} or GRID_LINEA_CHECKPOINT."
        )

    def _existing_path_from_env(self, name: str) -> str | None:
        raw_value = os.environ.get(name)
        if not raw_value:
            return None
        value = Path(raw_value)
        if value.exists():
            return str(value)
        return None

    def _resolve_linea_root(self) -> Path | None:
        for candidate in self._linea_root_candidates():
            if candidate.exists():
                return candidate
        return None

    def _linea_root_candidates(self) -> list[Path]:
        candidates = [Path("/opt/linea")]
        parents = Path(__file__).resolve().parents
        if len(parents) > 2:
            candidates.append(parents[2] / "LINEA")
        return candidates

    def _draw_grid_line(
        self,
        frame_bgr: np.ndarray,
        analysis_shape: tuple[int, int],
        orientation: str,
        line_state: dict[str, Any],
    ) -> None:
        analysis_h, analysis_w = analysis_shape
        render_h, render_w = frame_bgr.shape[:2]
        scale_x = float(render_w) / float(analysis_w)
        scale_y = float(render_h) / float(analysis_h)

        anchor, direction = self._line_anchor_and_direction(
            orientation,
            line_state,
            analysis_shape,
        )
        clipped = self._clip_infinite_line_to_frame(
            cx=anchor[0] * scale_x,
            cy=anchor[1] * scale_y,
            dx=direction[0] * scale_x,
            dy=direction[1] * scale_y,
            width=render_w,
            height=render_h,
        )
        if clipped is None:
            return

        cv2.line(
            frame_bgr,
            clipped[0],
            clipped[1],
            self._color,
            1,
            cv2.LINE_8,
        )

    def _build_viewport_corners(
        self,
        lines: dict[str, list[dict[str, Any]]],
        analysis_shape: tuple[int, int],
        render_shape: tuple[int, int],
    ) -> list[tuple[int, int]] | None:
        vertical = sorted(lines.get("vertical", []), key=lambda item: item["rho"])
        horizontal = sorted(lines.get("horizontal", []), key=lambda item: item["rho"])
        if len(vertical) < 2 or len(horizontal) < 2:
            return None

        left = vertical[0]
        right = vertical[-1]
        top = horizontal[0]
        bottom = horizontal[-1]

        corners_analysis = [
            self._intersect_lines("vertical", left, "horizontal", top, analysis_shape),
            self._intersect_lines("vertical", right, "horizontal", top, analysis_shape),
            self._intersect_lines("vertical", right, "horizontal", bottom, analysis_shape),
            self._intersect_lines("vertical", left, "horizontal", bottom, analysis_shape),
        ]
        if any(point is None for point in corners_analysis):
            return None

        analysis_h, analysis_w = analysis_shape
        for point in corners_analysis:
            if not self._point_is_reasonable(point, analysis_w, analysis_h):
                return None

        render_h, render_w = render_shape
        scale_x = float(render_w) / float(analysis_w)
        scale_y = float(render_h) / float(analysis_h)
        return [
            (
                int(round(point[0] * scale_x)),
                int(round(point[1] * scale_y)),
            )
            for point in corners_analysis
        ]

    def _intersect_lines(
        self,
        orientation_a: str,
        line_a: dict[str, Any],
        orientation_b: str,
        line_b: dict[str, Any],
        analysis_shape: tuple[int, int],
    ) -> tuple[float, float] | None:
        p1, d1 = self._line_anchor_and_direction(orientation_a, line_a, analysis_shape)
        p2, d2 = self._line_anchor_and_direction(orientation_b, line_b, analysis_shape)
        det = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(det) < 1e-6:
            return None

        diff_x = p2[0] - p1[0]
        diff_y = p2[1] - p1[1]
        t1 = (diff_x * d2[1] - diff_y * d2[0]) / det
        x = p1[0] + t1 * d1[0]
        y = p1[1] + t1 * d1[1]
        return (float(x), float(y))

    def _line_anchor_and_direction(
        self,
        orientation: str,
        line_state: dict[str, Any],
        analysis_shape: tuple[int, int],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        analysis_h, analysis_w = analysis_shape
        theta = float(line_state["theta"])
        if orientation == "vertical":
            anchor = (float(line_state["rho"]), analysis_h * 0.5)
        else:
            anchor = (analysis_w * 0.5, float(line_state["rho"]))
        direction = (float(np.cos(theta)), float(np.sin(theta)))
        return anchor, direction

    def _point_is_reasonable(self, point: tuple[float, float], width: int, height: int) -> bool:
        x, y = point
        return (-width) <= x <= (2 * width) and (-height) <= y <= (2 * height)

    def _clip_infinite_line_to_frame(
        self,
        cx: float,
        cy: float,
        dx: float,
        dy: float,
        width: int,
        height: int,
    ) -> tuple[tuple[int, int], tuple[int, int]] | None:
        eps = 1e-9
        points: list[tuple[float, float]] = []

        if abs(dx) > eps:
            t = (0.0 - cx) / dx
            y = cy + t * dy
            if 0.0 <= y <= height - 1:
                points.append((0.0, y))
            t = ((width - 1.0) - cx) / dx
            y = cy + t * dy
            if 0.0 <= y <= height - 1:
                points.append((width - 1.0, y))

        if abs(dy) > eps:
            t = (0.0 - cy) / dy
            x = cx + t * dx
            if 0.0 <= x <= width - 1:
                points.append((x, 0.0))
            t = ((height - 1.0) - cy) / dy
            x = cx + t * dx
            if 0.0 <= x <= width - 1:
                points.append((x, height - 1.0))

        unique: list[tuple[float, float]] = []
        for point in points:
            if not any(
                abs(point[0] - seen[0]) < 1e-4 and abs(point[1] - seen[1]) < 1e-4
                for seen in unique
            ):
                unique.append(point)

        if len(unique) < 2:
            return None

        unique.sort(key=lambda item: (item[0], item[1]))
        return (
            (int(round(unique[0][0])), int(round(unique[0][1]))),
            (int(round(unique[-1][0])), int(round(unique[-1][1]))),
        )
