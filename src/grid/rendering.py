from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .line import Line


class GridRenderer:
    def __init__(
        self,
        color_raw: tuple[int, int, int] = (0, 0, 255),
        color_rejected: tuple[int, int, int] = (0, 255, 0),
        color_predicted: tuple[int, int, int] = (255, 0, 255),
        color_accepted: tuple[int, int, int] = (255, 0, 255),
        color_flow: tuple[int, int, int] = (0, 165, 255),
        raw_dot_step_px: int = 5,
        rejected_dot_step_px: int = 5,
        predicted_dot_step_px: int = 8,
        accepted_dot_step_px: int = 8,
        flow_draw_arrow_scale: float = 6.0,
        flow_draw_arrow_min_px: int = 8,
        hist_panel_width: int = 72,
        hist_panel_margin: int = 12,
        color_hist_bar: tuple[int, int, int] = (210, 210, 210),
        color_hist_marker: tuple[int, int, int] = (255, 255, 255),
    ) -> None:
        self._dot_size_px = 6
        self._color_raw = color_raw
        self._color_rejected = color_rejected
        self._color_predicted = color_predicted
        self._color_accepted = color_accepted
        self._color_flow = color_flow
        self._color_hist_bar = color_hist_bar
        self._color_hist_marker = color_hist_marker
        self._raw_dot_step_px = raw_dot_step_px
        self._rejected_dot_step_px = rejected_dot_step_px
        self._predicted_dot_step_px = predicted_dot_step_px
        self._accepted_dot_step_px = accepted_dot_step_px
        self._flow_draw_arrow_scale = flow_draw_arrow_scale
        self._flow_draw_arrow_min_px = flow_draw_arrow_min_px
        self._hist_panel_width = max(int(hist_panel_width), 24)
        self._hist_panel_margin = max(int(hist_panel_margin), 0)

    def __call__(
        self,
        frame_bgr: np.ndarray,
        raw_lines: list[Line],
        grid_state: dict[str, dict[str, list[dict[str, Any]]]],
        analysis_shape: tuple[int, int],
        flow_shift: dict[str, float] | None = None,
        histogram_data: dict[str, Any] | None = None,
        accumulated_lines: list[Line] | None = None,
        show_flow_overlay: bool = True,
        show_legend: bool = True,
    ) -> np.ndarray:
        analysis_h, analysis_w = analysis_shape
        render_h, render_w = frame_bgr.shape[:2]
        scale_x = float(render_w) / float(analysis_w)
        scale_y = float(render_h) / float(analysis_h)
        flow_data = flow_shift or {"dx": 0.0, "dy": 0.0}
        left_reserved = 0
        self._draw_raw_lines(frame_bgr, raw_lines, scale_x, scale_y)
        if accumulated_lines:
            self._draw_raw_lines(frame_bgr, accumulated_lines, scale_x, scale_y)
        self._draw_state_group(
            frame_bgr,
            grid_state["rejected"],
            self._color_rejected,
            self._rejected_dot_step_px,
            analysis_shape,
            (scale_x, scale_y),
        )
        self._draw_state_group(
            frame_bgr,
            grid_state["predicted"],
            self._color_predicted,
            self._predicted_dot_step_px,
            analysis_shape,
            (scale_x, scale_y),
        )
        self._draw_state_group(
            frame_bgr,
            grid_state["accepted"],
            self._color_accepted,
            self._accepted_dot_step_px,
            analysis_shape,
            (scale_x, scale_y),
        )
        if histogram_data is not None:
            self._draw_debug_histograms(
                frame_bgr,
                histogram_data,
                grid_state["accepted"],
                analysis_shape,
                (scale_x, scale_y),
            )
        if show_flow_overlay:
            self._draw_flow_overlay(frame_bgr, flow_data)
        if show_legend:
            self._draw_legend(frame_bgr, left_reserved=left_reserved)
        return frame_bgr

    def _draw_raw_lines(
        self,
        frame_bgr: np.ndarray,
        raw_lines: list[Line],
        scale_x: float,
        scale_y: float,
    ) -> None:
        for item in raw_lines:
            self._draw_dotted_line(
                frame_bgr,
                (int(round(item.x1 * scale_x)), int(round(item.y1 * scale_y))),
                (int(round(item.x2 * scale_x)), int(round(item.y2 * scale_y))),
                color=self._color_raw,
                step_px=self._raw_dot_step_px,
                dot_size_px=2,
            )

    def _draw_state_group(
        self,
        frame_bgr: np.ndarray,
        groups: dict[str, list[dict[str, Any]]],
        color: tuple[int, int, int],
        step_px: int,
        analysis_shape: tuple[int, int],
        render_scale: tuple[float, float],
    ) -> None:
        for orientation in ("vertical", "horizontal"):
            for line in groups[orientation]:
                self._draw_state_line(frame_bgr, orientation, line, color, step_px, analysis_shape, render_scale)

    def _draw_state_line(
        self,
        frame_bgr: np.ndarray,
        orientation: str,
        line_state: dict[str, Any],
        color: tuple[int, int, int],
        step_px: int,
        analysis_shape: tuple[int, int],
        render_scale: tuple[float, float],
    ) -> None:
        analysis_h, analysis_w = analysis_shape
        scale_x, scale_y = render_scale
        if orientation == "vertical":
            cx, cy = line_state["rho"], analysis_h * 0.5
        else:
            cx, cy = analysis_w * 0.5, line_state["rho"]
        cx *= scale_x
        cy *= scale_y
        dx = math.cos(line_state["theta"]) * scale_x
        dy = math.sin(line_state["theta"]) * scale_y
        clipped = self._clip_infinite_line_to_frame(cx, cy, dx, dy, frame_bgr.shape[1], frame_bgr.shape[0])
        if clipped is None:
            return
        p1, p2 = clipped
        self._draw_dotted_line(frame_bgr, p1, p2, color=color, step_px=step_px)

    def _draw_flow_overlay(self, frame_bgr: np.ndarray, flow_shift: dict[str, float]) -> None:
        flow_dx = float(flow_shift.get("dx", 0.0))
        flow_dy = float(flow_shift.get("dy", 0.0))
        h, w = frame_bgr.shape[:2]
        panel_w = 170
        panel_h = 52
        x0 = max(12, w - panel_w - 12)
        y0 = 12
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0.0, frame_bgr)
        origin = (x0 + 20, y0 + panel_h // 2 + 2)
        arrow_dx = int(round(flow_dx * self._flow_draw_arrow_scale))
        arrow_dy = int(round(flow_dy * self._flow_draw_arrow_scale))
        if arrow_dx == 0 and arrow_dy == 0:
            arrow_dx = self._flow_draw_arrow_min_px
        arrow_end = (origin[0] + arrow_dx, origin[1] + arrow_dy)
        cv2.arrowedLine(
            frame_bgr,
            origin,
            arrow_end,
            self._color_flow,
            2,
            cv2.LINE_AA,
            tipLength=0.25,
        )
        cv2.circle(frame_bgr, origin, 2, self._color_flow, -1, cv2.LINE_AA)
        cv2.putText(
            frame_bgr,
            f"OF dx={flow_dx:+.2f}",
            (x0 + 48, y0 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame_bgr,
            f"dy={flow_dy:+.2f}",
            (x0 + 48, y0 + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    def _draw_legend(self, frame_bgr: np.ndarray, left_reserved: int = 0) -> None:
        legend_items = [
            ("raw", self._color_raw, self._raw_dot_step_px),
            ("gap-rejected", self._color_rejected, self._rejected_dot_step_px),
            ("tracker-pred", self._color_predicted, self._predicted_dot_step_px),
            ("accepted", self._color_accepted, self._accepted_dot_step_px),
        ]
        pad = 8
        row_h = 18
        sample_w = 52
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thickness = 1
        label_w = max(
            cv2.getTextSize(label, font, font_scale, font_thickness)[0][0]
            for label, _, _ in legend_items
        )
        panel_w = pad * 3 + sample_w + label_w
        panel_h = pad * 2 + row_h * len(legend_items)
        x0 = 12 + max(0, int(left_reserved))
        y0 = 12
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0.0, frame_bgr)
        for idx, (label, color, step_px) in enumerate(legend_items):
            y = y0 + pad + idx * row_h + row_h // 2
            x_line0 = x0 + pad
            x_line1 = x_line0 + sample_w
            self._draw_dotted_line(frame_bgr, (x_line0, y), (x_line1, y), color=color, step_px=step_px)
            cv2.putText(
                frame_bgr,
                label,
                (x_line1 + pad, y + 4),
                font,
                font_scale,
                (230, 230, 230),
                font_thickness,
                cv2.LINE_AA,
            )

    def _draw_debug_histograms(
        self,
        frame_bgr: np.ndarray,
        histogram_data: dict[str, Any],
        accepted_groups: dict[str, list[dict[str, Any]]],
        analysis_shape: tuple[int, int],
        render_scale: tuple[float, float],
    ) -> None:
        source_lines = histogram_data.get("source", [])
        self._draw_center_axes(frame_bgr, analysis_shape, render_scale)
        self._draw_vertical_axis_histogram_panel(
            frame_bgr,
            source_lines,
            analysis_shape,
            render_scale,
        )
        self._draw_horizontal_axis_histogram_panel(
            frame_bgr,
            source_lines,
            analysis_shape,
            render_scale,
        )
        self._draw_histogram_rho_guides(frame_bgr, accepted_groups, analysis_shape, render_scale)

    def _draw_center_axes(
        self,
        frame_bgr: np.ndarray,
        analysis_shape: tuple[int, int],
        render_scale: tuple[float, float],
    ) -> None:
        frame_h, frame_w = frame_bgr.shape[:2]
        center_x, center_y = self._center_axis_coords(analysis_shape, render_scale, frame_bgr.shape[:2])
        overlay = frame_bgr.copy()
        cv2.line(overlay, (center_x, 0), (center_x, frame_h - 1), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(overlay, (0, center_y), (frame_w - 1, center_y), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.30, frame_bgr, 0.70, 0.0, frame_bgr)

    def _draw_vertical_axis_histogram_panel(
        self,
        frame_bgr: np.ndarray,
        source_lines: list[Any],
        analysis_shape: tuple[int, int],
        render_scale: tuple[float, float],
    ) -> None:
        frame_h, frame_w = frame_bgr.shape[:2]
        panel_w = self._left_hist_panel_width(frame_w)
        center_x, _ = self._center_axis_coords(analysis_shape, render_scale, frame_bgr.shape[:2])
        x0, x1, anchor_x = self._vertical_hist_panel_bounds(center_x, frame_w, panel_w)
        y0 = 0
        y1 = frame_h - 1
        if panel_w <= 0 or y1 < y0 or x1 < x0:
            return
        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0.0, frame_bgr)

        bins = np.zeros(frame_h, dtype=np.float32)
        _, scale_y = render_scale
        for item in source_lines:
            if getattr(item, "orientation", None) != "horizontal":
                continue
            self._accumulate_projected_interval(
                bins,
                start_coord=float(getattr(item, "y1", item.axis_pos)) * scale_y,
                end_coord=float(getattr(item, "y2", item.axis_pos)) * scale_y,
                axis_span=frame_h,
                weight=max(float(getattr(item, "score", 0.0)), 1e-6),
            )

        peak = float(np.max(bins)) if bins.size > 0 else 0.0
        max_bar_len = max(1, x1 - x0 + 1)
        if peak > 0.0:
            for row, value in enumerate(bins):
                if value <= 0.0:
                    continue
                bar_len = max(1, int(round((value / peak) * max_bar_len)))
                y = y0 + row
                if anchor_x > center_x:
                    bar_x0 = x0
                    bar_x1 = min(x1, x0 + bar_len - 1)
                else:
                    bar_x1 = x1
                    bar_x0 = max(x0, x1 - bar_len + 1)
                cv2.line(
                    frame_bgr,
                    (bar_x0, y),
                    (bar_x1, y),
                    self._color_hist_bar,
                    1,
                    cv2.LINE_8,
                )
        cv2.line(frame_bgr, (anchor_x, y0), (anchor_x, y1), self._color_hist_marker, 1, cv2.LINE_AA)

    def _draw_horizontal_axis_histogram_panel(
        self,
        frame_bgr: np.ndarray,
        source_lines: list[Any],
        analysis_shape: tuple[int, int],
        render_scale: tuple[float, float],
    ) -> None:
        frame_h, frame_w = frame_bgr.shape[:2]
        panel_h = self._footer_hist_panel_height(frame_h)
        _, center_y = self._center_axis_coords(analysis_shape, render_scale, frame_bgr.shape[:2])
        x0 = 0
        x1 = frame_w - 1
        y0, y1, anchor_y = self._horizontal_hist_panel_bounds(center_y, frame_h, panel_h)
        if x1 < x0 or y1 < y0:
            return

        overlay = frame_bgr.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0.0, frame_bgr)

        bins = np.zeros(frame_w, dtype=np.float32)
        scale_x, _ = render_scale
        for item in source_lines:
            if getattr(item, "orientation", None) != "vertical":
                continue
            self._accumulate_projected_interval(
                bins,
                start_coord=float(getattr(item, "x1", item.axis_pos)) * scale_x,
                end_coord=float(getattr(item, "x2", item.axis_pos)) * scale_x,
                axis_span=frame_w,
                weight=max(float(getattr(item, "score", 0.0)), 1e-6),
            )

        peak = float(np.max(bins)) if bins.size > 0 else 0.0
        max_bar_h = max(1, y1 - y0 + 1)
        if peak > 0.0:
            for col, value in enumerate(bins):
                if value <= 0.0:
                    continue
                bar_h = max(1, int(round((value / peak) * max_bar_h)))
                x = x0 + col
                if anchor_y > center_y:
                    bar_y0 = max(y0, y1 - bar_h + 1)
                    bar_y1 = y1
                else:
                    bar_y0 = y0
                    bar_y1 = min(y1, y0 + bar_h - 1)
                cv2.line(
                    frame_bgr,
                    (x, bar_y0),
                    (x, bar_y1),
                    self._color_hist_bar,
                    1,
                    cv2.LINE_8,
                )
        cv2.line(frame_bgr, (x0, anchor_y), (x1, anchor_y), self._color_hist_marker, 1, cv2.LINE_AA)

    def _draw_histogram_rho_guides(
        self,
        frame_bgr: np.ndarray,
        accepted_groups: dict[str, list[dict[str, Any]]],
        analysis_shape: tuple[int, int],
        render_scale: tuple[float, float],
    ) -> None:
        frame_h, frame_w = frame_bgr.shape[:2]
        scale_x, scale_y = render_scale
        accepted = accepted_groups or {"vertical": [], "horizontal": []}
        center_x, center_y = self._center_axis_coords(analysis_shape, render_scale, frame_bgr.shape[:2])
        panel_w = self._left_hist_panel_width(frame_w)
        panel_h = self._footer_hist_panel_height(frame_h)
        vx0, vx1, v_anchor_x = self._vertical_hist_panel_bounds(center_x, frame_w, panel_w)
        hy0, hy1, h_anchor_y = self._horizontal_hist_panel_bounds(center_y, frame_h, panel_h)

        for line in accepted.get("horizontal", []):
            y = self._clamp_int(round(float(line["rho"]) * scale_y), 0, frame_h - 1)
            self._draw_dotted_line(
                frame_bgr,
                (min(v_anchor_x, center_x), y),
                (max(v_anchor_x, center_x), y),
                color=self._color_hist_marker,
                step_px=5,
                dot_size_px=1,
            )
            cv2.circle(
                frame_bgr,
                (center_x, y),
                2,
                self._color_accepted,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame_bgr,
                (v_anchor_x, y),
                2,
                self._color_accepted,
                -1,
                cv2.LINE_AA,
            )

        for line in accepted.get("vertical", []):
            x = self._clamp_int(round(float(line["rho"]) * scale_x), 0, frame_w - 1)
            self._draw_dotted_line(
                frame_bgr,
                (x, min(h_anchor_y, center_y)),
                (x, max(h_anchor_y, center_y)),
                color=self._color_hist_marker,
                step_px=5,
                dot_size_px=1,
            )
            cv2.circle(
                frame_bgr,
                (x, center_y),
                2,
                self._color_accepted,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame_bgr,
                (x, h_anchor_y),
                2,
                self._color_accepted,
                -1,
                cv2.LINE_AA,
            )

    def _left_hist_panel_width(self, frame_w: int) -> int:
        return max(1, min(frame_w, min(self._hist_panel_width, max(24, frame_w // 5))))

    def _footer_hist_panel_height(self, frame_h: int) -> int:
        return max(1, min(frame_h, min(self._hist_panel_width, max(24, frame_h // 5))))

    def _center_axis_coords(
        self,
        analysis_shape: tuple[int, int],
        render_scale: tuple[float, float],
        frame_shape: tuple[int, int],
    ) -> tuple[int, int]:
        frame_h, frame_w = frame_shape[:2]
        analysis_h, analysis_w = analysis_shape
        scale_x, scale_y = render_scale
        center_x = self._clamp_int(round((analysis_w * 0.5) * scale_x), 0, frame_w - 1)
        center_y = self._clamp_int(round((analysis_h * 0.5) * scale_y), 0, frame_h - 1)
        return center_x, center_y

    def _vertical_hist_panel_bounds(
        self,
        center_x: int,
        frame_w: int,
        panel_w: int,
    ) -> tuple[int, int, int]:
        preferred_x1 = center_x - self._hist_panel_margin - 1
        preferred_x0 = preferred_x1 - panel_w + 1
        if preferred_x0 >= 0:
            return preferred_x0, preferred_x1, preferred_x1

        x0 = min(max(center_x + self._hist_panel_margin + 1, 0), max(frame_w - panel_w, 0))
        x1 = min(frame_w - 1, x0 + panel_w - 1)
        return x0, x1, x0

    def _horizontal_hist_panel_bounds(
        self,
        center_y: int,
        frame_h: int,
        panel_h: int,
    ) -> tuple[int, int, int]:
        preferred_y0 = center_y + self._hist_panel_margin + 1
        preferred_y1 = preferred_y0 + panel_h - 1
        if preferred_y1 < frame_h:
            return preferred_y0, preferred_y1, preferred_y0

        y1 = max(center_y - self._hist_panel_margin - 1, 0)
        y0 = max(0, y1 - panel_h + 1)
        return y0, y1, y1

    def _hist_bin_index(self, rho: float, axis_span: int, bin_count: int) -> int:
        if bin_count <= 1 or axis_span <= 1:
            return 0
        rho_clamped = max(0.0, min(float(axis_span - 1), rho))
        scaled = rho_clamped * float(bin_count - 1) / float(axis_span - 1)
        return int(round(scaled))

    def _accumulate_projected_interval(
        self,
        bins: np.ndarray,
        start_coord: float,
        end_coord: float,
        axis_span: int,
        weight: float,
    ) -> None:
        if bins.size == 0:
            return
        start_idx = self._hist_bin_index(start_coord, axis_span, int(bins.size))
        end_idx = self._hist_bin_index(end_coord, axis_span, int(bins.size))
        lo = min(start_idx, end_idx)
        hi = max(start_idx, end_idx)
        count = max(1, hi - lo + 1)
        bins[lo : hi + 1] += float(weight) / float(count)

    def _clamp_int(self, value: int, low: int, high: int) -> int:
        return max(low, min(high, int(value)))

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
            if not any(abs(point[0] - seen[0]) < 1e-4 and abs(point[1] - seen[1]) < 1e-4 for seen in unique):
                unique.append(point)
        if len(unique) < 2:
            return None

        best_pair = None
        best_dist = -1.0
        for idx in range(len(unique)):
            for jdx in range(idx + 1, len(unique)):
                x1, y1 = unique[idx]
                x2, y2 = unique[jdx]
                dist = (x2 - x1) ** 2 + (y2 - y1) ** 2
                if dist > best_dist:
                    best_dist = dist
                    best_pair = (unique[idx], unique[jdx])
        if best_pair is None:
            return None

        (x1, y1), (x2, y2) = best_pair
        return (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2)))

    def _draw_dotted_line(
        self,
        image: np.ndarray,
        p1: tuple[int, int],
        p2: tuple[int, int],
        color: tuple[int, int, int],
        step_px: int,
        dot_size_px: int | None = None,
    ) -> None:
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)

        if length <= 0:
            if 0 <= x1 < image.shape[1] and 0 <= y1 < image.shape[0]:
                image[y1, x1] = color
            return

        step = 3.0 * max(1.0, float(step_px))
        dot_size = self._dot_size_px if dot_size_px is None else int(dot_size_px)
        dot_half = max(0, (dot_size - 1) // 2)
        ux = dx / length
        uy = dy / length
        dist = 0.0

        while dist <= length:
            px = int(round(x1 + ux * dist))
            py = int(round(y1 + uy * dist))
            if dot_half == 0:
                if 0 <= px < image.shape[1] and 0 <= py < image.shape[0]:
                    image[py, px] = color
            else:
                x0 = max(0, px - dot_half)
                x1b = min(image.shape[1], px + dot_half + 1)
                y0 = max(0, py - dot_half)
                y1b = min(image.shape[0], py + dot_half + 1)
                image[y0:y1b, x0:x1b] = color
            dist += step

        end_x = int(round(x2))
        end_y = int(round(y2))
        if 0 <= end_x < image.shape[1] and 0 <= end_y < image.shape[0]:
            image[end_y, end_x] = color
