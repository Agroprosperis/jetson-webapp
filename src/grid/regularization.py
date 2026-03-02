import math


class LineRegularizer:
    """Stabilize clustered line candidates by enforcing shared angle and spacing priors."""

    def __init__(
        self,
        parallel_ref_min_ratio: float = 0.70,
        parallel_ref_max_count: int = 3,
        parallel_force_ratio: float = 0.35,
        parallel_keep_tol_deg: float = 2.5,
        parallel_force_max_corr_deg: float = 1.5,
        spatial_reg_gain: float = 0.0,
        spatial_reg_max_corr: float = 2.5,
        gap_mode_tolerance_px: float = 5.0,
    ) -> None:
        """
        Initialize the line regularizer.

        Args:
            parallel_ref_min_ratio: Minimum confidence ratio, relative to the
                strongest line in the orientation, required for a line to be
                eligible as an angle reference.
            parallel_ref_max_count: Maximum number of strongest reference lines
                used to estimate the dominant orientation.
            parallel_force_ratio: Confidence ratio below which a line receives
                the full parallelization correction. Stronger lines still get a
                reduced correction so the grid remains coherent.
            parallel_keep_tol_deg: Maximum angular deviation, in degrees,
                allowed from the dominant orientation for a cluster to remain
                part of the grid family.
            parallel_force_max_corr_deg: Maximum angular correction, in
                degrees, applied when nudging a weak line toward the dominant
                orientation.
            spatial_reg_gain: Gain applied to the spacing correction before it
                is added to the line position.
            spatial_reg_max_corr: Maximum absolute position correction applied
                to a single line during one regularization pass.
            gap_mode_tolerance_px: Maximum deviation, in pixels, used to treat
                two observed gaps as part of the same gap family.
        """
        self._parallel_ref_min_ratio = parallel_ref_min_ratio
        self._parallel_ref_max_count = parallel_ref_max_count
        self._parallel_force_ratio = parallel_force_ratio
        self._parallel_keep_tol_rad = math.radians(max(parallel_keep_tol_deg, 0.0))
        self._parallel_force_max_corr_rad = math.radians(max(parallel_force_max_corr_deg, 0.0))
        self._spatial_reg_gain = spatial_reg_gain
        self._spatial_reg_max_corr = spatial_reg_max_corr
        self._gap_mode_tolerance_px = gap_mode_tolerance_px

    def __call__(self, clustered_lines: list[dict[str, any]]) -> list[dict[str, any]]:
        regularized_lines: list[dict[str, any]] = []
        for orientation in ("vertical", "horizontal"):
            orientation_lines = [line for line in clustered_lines if line["orientation"] == orientation]
            regularized_lines.extend(self._regularize_orientation(orientation_lines))
        return regularized_lines

    def _regularize_orientation(
        self,
        cluster_infos: list[dict[str, any]],
    ) -> list[dict[str, any]]:
        if not cluster_infos:
            return []
        
        regularized = [{
            "orientation": info["orientation"],
            "rho": float(info["rho"]),
            "theta": float(info["theta"]),
            "conf": float(info["conf"]),
        } for info in cluster_infos]

        dominant_theta = self._dominant_theta_from_clusters(self._parallel_reference_lines(regularized))
        filtered = self._filter_parallel_outliers(regularized, dominant_theta)
        if not filtered:
            filtered = regularized

        max_conf = max(info["conf"] for info in filtered)
        dominant_theta = self._dominant_theta_from_clusters(self._parallel_reference_lines(filtered))
        self._force_parallel_lines(filtered, dominant_theta, max_conf)
        return self._regularize_spacing(filtered)

    def _parallel_reference_lines(
        self,
        lines: list[dict[str, any]],
    ) -> list[dict[str, any]]:
        if not lines:
            return []
        max_conf = max(info["conf"] for info in lines)
        reference_lines = [i for i in lines if i["conf"] >= self._parallel_ref_min_ratio * max_conf]

        if not reference_lines:
            reference_lines = sorted(lines, key=lambda item: item["conf"], reverse=True)[:1]

        return sorted(reference_lines, key=lambda item: item["conf"], reverse=True)[: self._parallel_ref_max_count]

    def _filter_parallel_outliers(
        self,
        lines: list[dict[str, any]],
        dominant_theta: float,
    ) -> list[dict[str, any]]:
        if self._parallel_keep_tol_rad <= 0.0:
            return list(lines)
        filtered = [
            line
            for line in lines
            if abs(self._axis_angle_delta(line["theta"], dominant_theta)) <= self._parallel_keep_tol_rad
        ]
        return filtered

    def _force_parallel_lines(
        self,
        lines: list[dict[str, any]],
        dominant_theta: float,
        max_conf: float,
    ) -> None:
        if self._parallel_force_max_corr_rad <= 0.0:
            return
        for line in lines:
            conf_ratio = 1.0 if max_conf <= 1e-6 else float(line["conf"]) / float(max_conf)
            gain = self._parallel_correction_gain(conf_ratio)
            if gain <= 0.0:
                continue
            delta = self._axis_angle_delta(line["theta"], dominant_theta)
            limited_delta = self._clamp(
                delta,
                -self._parallel_force_max_corr_rad,
                self._parallel_force_max_corr_rad,
            )
            line["theta"] += gain * limited_delta

    def _regularize_spacing(
        self,
        lines: list[dict[str, any]],
    ) -> list[dict[str, any]]:
        if self._spatial_reg_gain <= 0.0 or self._spatial_reg_max_corr <= 0.0:
            return sorted(lines, key=lambda item: item["rho"])
        gap_modes = self._estimate_narrow_wide_gap_modes(self._collect_neighbor_gaps(lines))
        
        if not gap_modes or len(lines) < 3:
            return sorted(lines, key=lambda item: item["rho"])
        
        ordered = sorted(lines, key=lambda item: item["rho"])
        
        for idx, line in enumerate(ordered):
            correction = self._compute_spacing_correction(ordered, idx, gap_modes)
            line["rho"] += self._clamp(
                correction * self._spatial_reg_gain,
                -self._spatial_reg_max_corr,
                self._spatial_reg_max_corr,
            )
        return ordered

    def _compute_spacing_correction(
        self,
        ordered_lines: list[dict[str, any]],
        index: int,
        gap_modes: list[float],
    ) -> float:
        correction = 0.0
        current_line = ordered_lines[index]

        if index > 0:
            left_gap = current_line["rho"] - ordered_lines[index - 1]["rho"]
            target = min(gap_modes, key=lambda mode: abs(mode - left_gap))
            correction -= left_gap - target

        if index + 1 < len(ordered_lines):
            right_gap = ordered_lines[index + 1]["rho"] - current_line["rho"]
            target = min(gap_modes, key=lambda mode: abs(mode - right_gap))
            correction += right_gap - target

        return correction

    def _dominant_theta_from_clusters(
        self,
        cluster_infos: list[dict[str, any]],
    ) -> float:
        c2 = 0.0
        s2 = 0.0
        for info in cluster_infos:
            weight = max(info["conf"], 1e-6)
            c2 += weight * math.cos(2.0 * info["theta"])
            s2 += weight * math.sin(2.0 * info["theta"])
        return 0.5 * math.atan2(s2, c2)

    def _collect_neighbor_gaps(
        self,
        lines: list[dict[str, any]],
    ) -> list[float]:
        ordered = sorted(lines, key=lambda line: line["rho"])
        gaps: list[float] = []
        for idx in range(len(ordered) - 1):
            gap = float(ordered[idx + 1]["rho"] - ordered[idx]["rho"])
            if gap > 1.5:
                gaps.append(gap)
        return gaps

    def _find_best_gap_cluster(
        self,
        gaps: list[float],
    ) -> dict[str, float] | None:
        ordered = sorted(float(gap) for gap in gaps if gap > 0.0)
        if not ordered:
            return None

        best = None
        count = len(ordered)
        for start in range(count):
            for end in range(start, count):
                cluster = ordered[start : end + 1]
                center = sum(cluster) / len(cluster)
                if max(abs(gap - center) for gap in cluster) > self._gap_mode_tolerance_px:
                    continue
                candidate = {
                    "center": center,
                    "count": len(cluster),
                    "err": sum(abs(gap - center) for gap in cluster),
                }
                if best is None:
                    best = candidate
                    continue
                if candidate["count"] > best["count"]:
                    best = candidate
                    continue
                if candidate["count"] == best["count"] and candidate["err"] < best["err"] - 1e-6:
                    best = candidate
                    continue
                if (
                    candidate["count"] == best["count"]
                    and abs(candidate["err"] - best["err"]) <= 1e-6
                    and candidate["center"] > best["center"]
                ):
                    best = candidate
        return best

    def _estimate_narrow_wide_gap_modes(
        self,
        gaps: list[float],
    ) -> list[float]:
        ordered = sorted(float(gap) for gap in gaps if gap > 0.0)
        if not ordered:
            return []
        if len(ordered) == 1:
            return [ordered[0]]

        primary = self._find_best_gap_cluster(ordered)
        if primary is None:
            return []

        primary_center = primary["center"]
        larger_side = [gap for gap in ordered if gap > primary_center + self._gap_mode_tolerance_px]
        smaller_side = [gap for gap in ordered if gap < primary_center - self._gap_mode_tolerance_px]

        secondary = None
        if larger_side:
            secondary = self._find_best_gap_cluster(larger_side)
        if secondary is None and smaller_side:
            secondary = self._find_best_gap_cluster(smaller_side)

        modes = [primary_center]
        if secondary is not None:
            modes.append(secondary["center"])
        elif larger_side:
            modes.append(max(larger_side))

        modes = sorted(float(mode) for mode in modes if mode > 0.0)
        if len(modes) >= 2 and abs(modes[1] - modes[0]) <= self._gap_mode_tolerance_px:
            return [sum(modes) / len(modes)]
        return modes

    def _axis_angle_delta(
        self,
        theta_from: float,
        theta_to: float,
    ) -> float:
        return 0.5 * math.atan2(
            math.sin(2.0 * (theta_to - theta_from)),
            math.cos(2.0 * (theta_to - theta_from)),
        )

    def _parallel_correction_gain(self, conf_ratio: float) -> float:
        conf_ratio = self._clamp(conf_ratio, 0.0, 1.0)
        threshold = self._clamp(self._parallel_force_ratio, 0.0, 1.0)
        if conf_ratio <= threshold:
            return 1.0
        span = max(1e-6, 1.0 - threshold)
        taper = 1.0 - (conf_ratio - threshold) / span
        return max(0.5, taper)

    def _clamp(
        self,
        value: float,
        low: float,
        high: float,
    ) -> float:
        return max(low, min(high, value))
