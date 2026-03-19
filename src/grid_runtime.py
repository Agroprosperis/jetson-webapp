from __future__ import annotations

from collections import deque
import logging
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from grid import GridRenderer

LOGGER = logging.getLogger("grid_runtime")

DEFAULT_GRID_COLOR = (0, 255, 255)
DEFAULT_LINEA_CONFIG = "configs/linea/linea_hgnetv2_s.py"
DEFAULT_LINEA_CHECKPOINT = "weights/linea_hgnetv2_s.pth"
DEFAULT_APP_CHECKPOINT = "/app/model/linea/linea_hgnetv2_s.pth"
FIXED_GAP_PRIOR_RATIOS = {
    "vertical": (0.0500, 0.2030),
    "horizontal": (0.0667, 0.2707),
}


@dataclass(slots=True)
class GridOverlay:
    analysis_shape: tuple[int, int]
    lines: dict[str, list[dict[str, Any]]]
    viewport_corners: list[tuple[int, int]] | None
    score: float | None
    score_raw: float | None
    clustered_lines: dict[str, list[dict[str, Any]]] | None = None
    selected_lines: dict[str, list[dict[str, Any]]] | None = None
    raw_lines: list[Any] | None = None
    accumulated_lines: list[Any] | None = None
    grid_state: dict[str, dict[str, list[dict[str, Any]]]] | None = None
    flow_shift: dict[str, float] | None = None
    histogram_data: dict[str, Any] | None = None


class GridProcessor:
    def __init__(
        self,
        color: tuple[int, int, int] = DEFAULT_GRID_COLOR,
        cluster_only_mode: bool = False,
        preprocess_enabled: bool = False,
        orientation_prior_ema_alpha: float = 0.20,
        orientation_prior_keep_tol_deg: float = 1.25,
        orientation_prior_update_tol_deg: float = 0.625,
        orientation_prior_ref_min_ratio: float = 0.70,
        orientation_prior_ref_max_count: int = 10,
        orientation_prior_seed_tol_deg: float = 0.75,
    ) -> None:
        self._color = color
        self._cluster_only_mode = bool(cluster_only_mode)
        self._preprocess_enabled = bool(preprocess_enabled)
        self._orientation_prior_ema_alpha = min(max(float(orientation_prior_ema_alpha), 0.0), 1.0)
        self._orientation_prior_keep_tol_rad = math.radians(max(float(orientation_prior_keep_tol_deg), 0.0))
        self._orientation_prior_update_tol_rad = math.radians(max(float(orientation_prior_update_tol_deg), 0.0))
        self._orientation_prior_ref_min_ratio = max(float(orientation_prior_ref_min_ratio), 0.0)
        self._orientation_prior_ref_max_count = max(int(orientation_prior_ref_max_count), 1)
        self._orientation_prior_seed_tol_rad = math.radians(max(float(orientation_prior_seed_tol_deg), 0.0))
        self._debug_renderer = GridRenderer()
        self._processed_frames = 0
        self._last_gap_info: dict[str, list[float]] = {
            "vertical": [],
            "horizontal": [],
        }
        self._viewport_bounce_ratio_threshold = 1.50
        self._viewport_bounce_window_sec = 3.0
        self._viewport_bounce_max_events = 5
        self._viewport_bounce_events: deque[float] = deque()
        self._last_viewport_size: tuple[float, float] | None = None
        self._last_viewport_score: float | None = None
        self._orientation_priors: dict[str, float | None] = {
            "vertical": None,
            "horizontal": None,
        }
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

    def process(self, frame_bgr: np.ndarray, *, include_debug: bool = False) -> GridOverlay | None:
        if self._processing_failed:
            return None
        if not self._ensure_initialized():
            return None

        try:
            if self._preprocess is not None:
                analysis_rgb = self._preprocess(frame_bgr)
            else:
                analysis_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            analysis_shape = analysis_rgb.shape[:2]
            if self._tracker is not None:
                analysis_gray = cv2.cvtColor(analysis_rgb, cv2.COLOR_RGB2GRAY)
                flow_shift = self._tracker.estimate_flow_shift(analysis_gray)
            else:
                flow_shift = None
            self._processed_frames += 1
            raw_lines = self._detector(analysis_rgb)
            cluster_input_lines, orientation_prior_debug = self._apply_orientation_priors(raw_lines)
            accumulated_lines: list[Any] = []
            histogram_data: dict[str, Any] | None = None

            if self._cluster_only_mode:
                clustered_lines = self._clusterizer(cluster_input_lines, analysis_shape)
                vertical_clustered_count = sum(1 for line in clustered_lines if line["orientation"] == "vertical")
                horizontal_clustered_count = sum(1 for line in clustered_lines if line["orientation"] == "horizontal")
                gap_info = self._fixed_gap_info(analysis_shape)
                self._last_gap_info = {
                    "vertical": list(gap_info["vertical"]),
                    "horizontal": list(gap_info["horizontal"]),
                }
                current_builder = self._grid_builder_cls() if self._grid_builder_cls is not None else self._grid_builder
                current_grid = current_builder.build_current(clustered_lines, gap_info)
                LOGGER.info(
                    "Grid selection debug frame=%d v_acc=%d/%d h_acc=%d/%d v_acc_gaps=%s h_acc_gaps=%s learned_v=%s learned_h=%s",
                    self._processed_frames,
                    len(current_grid["accepted"]["vertical"]),
                    vertical_clustered_count,
                    len(current_grid["accepted"]["horizontal"]),
                    horizontal_clustered_count,
                    self._neighbor_gaps_from_public_lines(current_grid["accepted"]["vertical"]),
                    self._neighbor_gaps_from_public_lines(current_grid["accepted"]["horizontal"]),
                    self._last_gap_info["vertical"],
                    self._last_gap_info["horizontal"],
                )
                LOGGER.info(
                    "Grid orientation priors frame=%d v_prior=%s v_frame=%s v_update=%d v_keep=%d/%d h_prior=%s h_frame=%s h_update=%d h_keep=%d/%d",
                    self._processed_frames,
                    orientation_prior_debug["vertical"]["prior_deg"],
                    orientation_prior_debug["vertical"]["frame_deg"],
                    orientation_prior_debug["vertical"]["update_count"],
                    orientation_prior_debug["vertical"]["kept_count"],
                    orientation_prior_debug["vertical"]["total_count"],
                    orientation_prior_debug["horizontal"]["prior_deg"],
                    orientation_prior_debug["horizontal"]["frame_deg"],
                    orientation_prior_debug["horizontal"]["update_count"],
                    orientation_prior_debug["horizontal"]["kept_count"],
                    orientation_prior_debug["horizontal"]["total_count"],
                )
            else:
                accumulated_lines = self._accumulator(cluster_input_lines, flow_shift)
                clustered_lines = self._clusterizer(accumulated_lines, analysis_shape)
                regularized_lines = self._regularizer(clustered_lines)
                gap_info = self._fixed_gap_info(analysis_shape)
                current_grid = self._grid_builder.build_current(regularized_lines, gap_info)
                self._last_gap_info = {
                    "vertical": list(gap_info["vertical"]),
                    "horizontal": list(gap_info["horizontal"]),
                }

            if self._tracker is not None:
                tracker_input = current_grid["accepted"]["vertical"] + current_grid["accepted"]["horizontal"]
                tracker_state = self._tracker(tracker_input, flow_shift=flow_shift)
                grid_state = self._grid_builder.integrate_predictions(current_grid, gap_info, tracker_state)
            elif self._cluster_only_mode:
                grid_state = self._current_only_grid_state(current_grid)
            else:
                grid_state = self._current_only_grid_state(current_grid)
            overlay_lines = self._overlay_lines_from_grid_state(grid_state)
            overlay_lines = self._filter_lines_within_frame(overlay_lines, analysis_shape)
            if include_debug:
                histogram_source = accumulated_lines if accumulated_lines else cluster_input_lines
                histogram_data = {"source": list(histogram_source)}

            viewport_corners = self._build_viewport_corners(
                overlay_lines,
                analysis_shape,
                frame_bgr.shape[:2],
            )
            viewport_score = self._update_viewport_bounce_score(viewport_corners)
            if include_debug and viewport_score is not None:
                LOGGER.info(
                    "Grid viewport stability frame=%d score=%.3f bounce_events=%d viewport_size=%s",
                    self._processed_frames,
                    viewport_score,
                    len(self._viewport_bounce_events),
                    self._last_viewport_size,
                )
            return GridOverlay(
                analysis_shape=analysis_shape,
                lines=overlay_lines,
                viewport_corners=viewport_corners,
                score=viewport_score,
                score_raw=None,
                clustered_lines=self._group_public_lines(clustered_lines) if include_debug else None,
                selected_lines=self._current_only_grid_state(current_grid)["accepted"] if include_debug else None,
                raw_lines=list(raw_lines) if include_debug else None,
                accumulated_lines=list(accumulated_lines) if include_debug else None,
                grid_state=grid_state if include_debug else None,
                flow_shift=dict(flow_shift) if include_debug and flow_shift is not None else None,
                histogram_data=histogram_data,
            )
        except Exception:
            self._processing_failed = True
            LOGGER.exception("Grid processing failed. Disabling grid detection for this run.")
            return None

    def render(self, frame_bgr: np.ndarray, overlay: GridOverlay | None, *, debug: bool = False) -> np.ndarray:
        if overlay is None:
            return frame_bgr

        if debug and overlay.grid_state is not None:
            frame_bgr = self._debug_renderer(
                frame_bgr,
                clustered_groups=overlay.clustered_lines or {"vertical": [], "horizontal": []},
                selected_groups=overlay.selected_lines or {"vertical": [], "horizontal": []},
                accumulated_lines=overlay.accumulated_lines,
                grid_state=overlay.grid_state,
                analysis_shape=overlay.analysis_shape,
                flow_shift=overlay.flow_shift,
                histogram_data=overlay.histogram_data,
                show_flow_overlay=overlay.flow_shift is not None,
            )
        else:
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
                    3,
                    cv2.LINE_8,
                )

        return frame_bgr

    def reset_state(self, *, preserve_score: bool = True) -> None:
        self._processed_frames = 0
        self._last_gap_info = {
            "vertical": [],
            "horizontal": [],
        }
        self._viewport_bounce_events.clear()
        self._last_viewport_size = None
        if not preserve_score:
            self._last_viewport_score = None
        if self._tracker_cls is not None:
            self._tracker = self._tracker_cls()
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

            self._preprocess = FramePreprocessor() if self._preprocess_enabled else None
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
            self.reset_state()
            self._initialized = True
            LOGGER.info(
                "Grid processor initialized with config=%s checkpoint=%s cluster_only_mode=%s preprocess_enabled=%s",
                config_path,
                checkpoint_path,
                self._cluster_only_mode,
                self._preprocess_enabled,
            )
            return True
        except Exception:
            self._init_failed = True
            LOGGER.exception("Grid processor initialization failed.")
            return False

    def _empty_grid_state(self) -> dict[str, dict[str, list[dict[str, Any]]]]:
        return {
            "accepted": {"vertical": [], "horizontal": []},
            "rejected": {"vertical": [], "horizontal": []},
        }

    def _apply_orientation_priors(
        self,
        lines: list[Any],
    ) -> tuple[list[Any], dict[str, dict[str, Any]]]:
        kept: list[Any] = []
        debug: dict[str, dict[str, Any]] = {}

        for orientation in ("vertical", "horizontal"):
            orientation_lines = [line for line in lines if getattr(line, "orientation", None) == orientation]
            prior_theta = self._orientation_priors[orientation]
            update_lines = self._orientation_prior_update_lines(orientation_lines, prior_theta)
            frame_theta = self._estimate_orientation_theta(update_lines)

            if frame_theta is not None:
                if prior_theta is None:
                    prior_theta = frame_theta
                else:
                    prior_theta = self._angle_blend(
                        float(prior_theta),
                        float(frame_theta),
                        self._orientation_prior_ema_alpha,
                    )
                self._orientation_priors[orientation] = prior_theta

            if prior_theta is None or self._orientation_prior_keep_tol_rad <= 0.0:
                kept_lines = list(orientation_lines)
            else:
                kept_lines = [
                    line
                    for line in orientation_lines
                    if abs(self._axis_angle_delta(float(getattr(line, "theta", 0.0)), float(prior_theta)))
                    <= self._orientation_prior_keep_tol_rad
                ]

            kept.extend(kept_lines)
            debug[orientation] = {
                "prior_deg": self._theta_deg(prior_theta),
                "frame_deg": self._theta_deg(frame_theta),
                "update_count": len(update_lines),
                "kept_count": len(kept_lines),
                "total_count": len(orientation_lines),
            }

        return kept, debug

    def _estimate_orientation_theta(self, lines: list[Any]) -> float | None:
        if not lines:
            return None

        strongest_score = max(max(float(getattr(line, "score", 0.0)), 0.0) for line in lines)
        if strongest_score <= 0.0:
            reference_lines = list(lines[:1])
        else:
            min_score = self._orientation_prior_ref_min_ratio * strongest_score
            reference_lines = [
                line
                for line in lines
                if max(float(getattr(line, "score", 0.0)), 0.0) >= min_score
            ]
            if not reference_lines:
                reference_lines = [max(lines, key=lambda item: float(getattr(item, "score", 0.0)))]

        reference_lines = sorted(
            reference_lines,
            key=lambda item: float(getattr(item, "score", 0.0)),
            reverse=True,
        )[: self._orientation_prior_ref_max_count]

        best_cluster: list[Any] | None = None
        best_weight = -1.0
        best_error = float("inf")

        for seed in reference_lines:
            seed_theta = float(getattr(seed, "theta", 0.0))
            cluster = [
                line
                for line in reference_lines
                if abs(self._axis_angle_delta(float(getattr(line, "theta", 0.0)), seed_theta))
                <= self._orientation_prior_seed_tol_rad
            ]
            if not cluster:
                continue
            cluster_theta = self._weighted_theta(cluster)
            cluster_weight = sum(max(float(getattr(line, "score", 0.0)), 1e-6) for line in cluster)
            cluster_error = sum(
                max(float(getattr(line, "score", 0.0)), 1e-6)
                * abs(self._axis_angle_delta(float(getattr(line, "theta", 0.0)), cluster_theta))
                for line in cluster
            )
            if cluster_weight > best_weight + 1e-6:
                best_cluster = cluster
                best_weight = cluster_weight
                best_error = cluster_error
                continue
            if abs(cluster_weight - best_weight) <= 1e-6 and cluster_error < best_error - 1e-6:
                best_cluster = cluster
                best_error = cluster_error

        if best_cluster is None:
            best_cluster = reference_lines

        return self._weighted_theta(best_cluster)

    def _orientation_prior_update_lines(
        self,
        lines: list[Any],
        prior_theta: float | None,
    ) -> list[Any]:
        if prior_theta is None or self._orientation_prior_update_tol_rad <= 0.0:
            return list(lines)
        return [
            line
            for line in lines
            if abs(self._axis_angle_delta(float(getattr(line, "theta", 0.0)), float(prior_theta)))
            <= self._orientation_prior_update_tol_rad
        ]

    def _weighted_theta(self, lines: list[Any]) -> float:
        c2 = 0.0
        s2 = 0.0
        for line in lines:
            weight = max(float(getattr(line, "score", 0.0)), 1e-6)
            theta = float(getattr(line, "theta", 0.0))
            c2 += weight * math.cos(2.0 * theta)
            s2 += weight * math.sin(2.0 * theta)
        return 0.5 * math.atan2(s2, c2)

    def _axis_angle_delta(
        self,
        theta_from: float,
        theta_to: float,
    ) -> float:
        return 0.5 * math.atan2(
            math.sin(2.0 * (theta_to - theta_from)),
            math.cos(2.0 * (theta_to - theta_from)),
        )

    def _angle_blend(
        self,
        theta_old: float,
        theta_new: float,
        alpha: float,
    ) -> float:
        c_old = math.cos(2.0 * theta_old)
        s_old = math.sin(2.0 * theta_old)
        c_new = math.cos(2.0 * theta_new)
        s_new = math.sin(2.0 * theta_new)
        c_mix = (1.0 - alpha) * c_old + alpha * c_new
        s_mix = (1.0 - alpha) * s_old + alpha * s_new
        return 0.5 * math.atan2(s_mix, c_mix)

    def _theta_deg(self, theta: float | None) -> str:
        if theta is None:
            return "-"
        return f"{(abs(math.degrees(float(theta))) % 180.0):.2f}"

    def _neighbor_gaps_from_public_lines(
        self,
        lines: list[dict[str, Any]],
    ) -> list[float]:
        ordered = sorted(float(line["rho"]) for line in lines)
        return [
            round(ordered[idx + 1] - ordered[idx], 2)
            for idx in range(len(ordered) - 1)
            if (ordered[idx + 1] - ordered[idx]) > 1.5
        ]

    def _fixed_gap_info(
        self,
        analysis_shape: tuple[int, int],
    ) -> dict[str, list[float]]:
        analysis_h, analysis_w = analysis_shape
        vertical_small, vertical_big = FIXED_GAP_PRIOR_RATIOS["vertical"]
        horizontal_small, horizontal_big = FIXED_GAP_PRIOR_RATIOS["horizontal"]
        return {
            "vertical": [
                round(vertical_small * float(analysis_w), 3),
                round(vertical_big * float(analysis_w), 3),
            ],
            "horizontal": [
                round(horizontal_small * float(analysis_h), 3),
                round(horizontal_big * float(analysis_h), 3),
            ],
        }

    def _group_public_lines(
        self,
        lines: list[dict[str, Any]],
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "vertical": sorted(
                [dict(line) for line in lines if line["orientation"] == "vertical"],
                key=lambda item: item["rho"],
            ),
            "horizontal": sorted(
                [dict(line) for line in lines if line["orientation"] == "horizontal"],
                key=lambda item: item["rho"],
            ),
        }

    def _overlay_lines_from_grid_state(
        self,
        grid_state: dict[str, dict[str, list[dict[str, Any]]]],
    ) -> dict[str, list[dict[str, Any]]]:
        return {
            "vertical": list(grid_state["accepted"]["vertical"]),
            "horizontal": list(grid_state["accepted"]["horizontal"]),
        }

    def _current_only_grid_state(
        self,
        current_grid: dict[str, dict[str, list[dict[str, Any]]]],
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        return {
            "accepted": {
                "vertical": [dict(line) for line in current_grid["accepted"]["vertical"]],
                "horizontal": [dict(line) for line in current_grid["accepted"]["horizontal"]],
            },
            "rejected": {
                "vertical": [dict(line) for line in current_grid["rejected"]["vertical"]],
                "horizontal": [dict(line) for line in current_grid["rejected"]["horizontal"]],
            },
            "predicted": {"vertical": [], "horizontal": []},
        }

    def _clustered_only_grid_state(
        self,
        clustered_lines: list[dict[str, Any]],
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        accepted = {
            "vertical": sorted(
                [dict(line) for line in clustered_lines if line["orientation"] == "vertical"],
                key=lambda item: item["rho"],
            ),
            "horizontal": sorted(
                [dict(line) for line in clustered_lines if line["orientation"] == "horizontal"],
                key=lambda item: item["rho"],
            ),
        }
        return {
            "accepted": accepted,
            "rejected": {"vertical": [], "horizontal": []},
            "predicted": {"vertical": [], "horizontal": []},
        }

    def _tracking_only_overlay_lines(
        self,
        current_grid: dict[str, dict[str, list[dict[str, Any]]]],
        gap_info: dict[str, list[float]],
        tracker_state: dict[str, Any],
    ) -> dict[str, list[dict[str, Any]]]:
        grid_state = self._tracking_only_grid_state(current_grid, gap_info, tracker_state)
        return self._overlay_lines_from_grid_state(grid_state)

    def _tracking_only_grid_state(
        self,
        current_grid: dict[str, dict[str, list[dict[str, Any]]]],
        gap_info: dict[str, list[float]],
        tracker_state: dict[str, Any],
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
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
            return {
                "accepted": reference_lines,
                "rejected": {
                    "vertical": [dict(line) for line in current_grid["rejected"]["vertical"]],
                    "horizontal": [dict(line) for line in current_grid["rejected"]["horizontal"]],
                },
                "predicted": {
                    "vertical": sorted(
                        [dict(line) for line in tracker_state["predicted"]["vertical"]],
                        key=lambda item: item["rho"],
                    ),
                    "horizontal": sorted(
                        [dict(line) for line in tracker_state["predicted"]["horizontal"]],
                        key=lambda item: item["rho"],
                    ),
                },
            }

        return self._grid_builder.integrate_predictions(current_grid, gap_info, tracker_state)

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

    def _update_viewport_bounce_score(
        self,
        viewport_corners: list[tuple[int, int]] | None,
    ) -> float | None:
        now = time.monotonic()
        self._prune_viewport_bounce_events(now)
        current_size = self._viewport_size(viewport_corners)
        if current_size is None:
            self._last_viewport_size = None
            if self._last_viewport_score is None:
                return None
            score = self._viewport_score_from_bounce_count(len(self._viewport_bounce_events))
            self._last_viewport_score = score
            return score

        previous_size = self._last_viewport_size
        if previous_size is not None:
            prev_width, prev_height = previous_size
            curr_width, curr_height = current_size
            if (
                self._dimension_bounced(prev_width, curr_width)
                or self._dimension_bounced(prev_height, curr_height)
            ):
                self._viewport_bounce_events.append(now)
                self._prune_viewport_bounce_events(now)

        self._last_viewport_size = current_size
        score = self._viewport_score_from_bounce_count(len(self._viewport_bounce_events))
        self._last_viewport_score = score
        return score

    def _prune_viewport_bounce_events(self, now: float) -> None:
        min_time = now - self._viewport_bounce_window_sec
        while self._viewport_bounce_events and self._viewport_bounce_events[0] < min_time:
            self._viewport_bounce_events.popleft()

    def _viewport_score_from_bounce_count(self, bounce_count: int) -> float:
        if self._viewport_bounce_max_events <= 0:
            return 1.0
        clamped_count = min(max(int(bounce_count), 0), self._viewport_bounce_max_events)
        return max(0.0, 1.0 - (float(clamped_count) / float(self._viewport_bounce_max_events)))

    def _dimension_bounced(self, previous: float, current: float) -> bool:
        if previous <= 1e-6 or current <= 1e-6:
            return False
        return abs(current - previous) / previous > self._viewport_bounce_ratio_threshold

    def _viewport_size(
        self,
        viewport_corners: list[tuple[int, int]] | None,
    ) -> tuple[float, float] | None:
        if viewport_corners is None or len(viewport_corners) != 4:
            return None
        points = [np.asarray(point, dtype=np.float32) for point in viewport_corners]
        top = float(np.linalg.norm(points[1] - points[0]))
        right = float(np.linalg.norm(points[2] - points[1]))
        bottom = float(np.linalg.norm(points[2] - points[3]))
        left = float(np.linalg.norm(points[3] - points[0]))
        width = 0.5 * (top + bottom)
        height = 0.5 * (left + right)
        if width <= 1e-6 or height <= 1e-6:
            return None
        return (width, height)

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
