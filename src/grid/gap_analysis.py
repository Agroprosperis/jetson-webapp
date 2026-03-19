class GapAnalyzer:
    """Maintain a single running estimate of allowed gap intervals per orientation."""

    def __init__(
        self,
        ema: float = 0.25,
        gap_count: int = 2,
        gap_tolerance_px: float = 5.0,
        max_gap_multiple: int = 4,
    ) -> None:
        """
        Initialize the gap analyzer.

        Args:
            ema: Exponential moving average factor used to update stored gaps
                with the current frame estimate.
            gap_count: Number of gap intervals to track per orientation.
            gap_tolerance_px: Maximum deviation, in pixels, used when grouping
                observed gaps into the same interval family.
            max_gap_multiple: Maximum integer multiplier allowed when an
                observed neighbor gap spans one or more missed lines.
        """
        self._ema = ema
        self._gap_count = gap_count
        self._gap_tolerance_px = gap_tolerance_px
        self._max_gap_multiple = max(int(max_gap_multiple), 1)
        self._gaps: dict[str, list[float]] = {
            "vertical": [],
            "horizontal": [],
        }

    def __call__(self, clustered_lines: list[dict[str, any]]) -> dict[str, list[float]]:
        result: dict[str, list[float]] = {}
        
        for orientation in ("vertical", "horizontal"):
            lines = [line for line in clustered_lines if line["orientation"] == orientation]
            frame_gaps = self._estimate_frame_gaps(lines, self._gaps[orientation])
            self._gaps[orientation] = self._update_gaps(self._gaps[orientation], frame_gaps)
            result[orientation] = list(self._gaps[orientation])

        return result

    def _estimate_frame_gaps(
        self,
        lines: list[dict[str, any]],
        reference_gaps: list[float],
    ) -> list[float]:
        observed_gaps = self._collect_neighbor_gaps(lines)
        if not observed_gaps or self._gap_count <= 0:
            return []
        
        if reference_gaps:
            return self._estimate_gaps_from_reference(observed_gaps, reference_gaps)

        primary_cluster = self._find_best_gap_cluster(observed_gaps)
        if primary_cluster is None:
            return []

        estimated_gaps = [float(primary_cluster["center"])]
        if self._gap_count == 1:
            return estimated_gaps

        larger_gaps = [
            gap
            for gap in observed_gaps
            if gap > primary_cluster["center"] + self._gap_tolerance_px
        ]
        smaller_gaps = [
            gap
            for gap in observed_gaps
            if gap < primary_cluster["center"] - self._gap_tolerance_px
        ]
        secondary_cluster = self._find_best_gap_cluster(larger_gaps)
        if secondary_cluster is None:
            secondary_cluster = self._find_best_gap_cluster(smaller_gaps)
        if secondary_cluster is not None:
            estimated_gaps.append(float(secondary_cluster["center"]))

        return sorted(estimated_gaps)

    def _estimate_gaps_from_reference(
        self,
        observed_gaps: list[float],
        reference_gaps: list[float],
    ) -> list[float]:
        estimated_gaps: list[float] = []
        ordered_gaps = sorted(float(gap) for gap in observed_gaps)
        for reference_gap in sorted(float(gap) for gap in reference_gaps):
            candidates = [
                normalized_gap
                for gap in ordered_gaps
                for normalized_gap in [self._normalize_gap_to_reference(gap, reference_gap)]
                if normalized_gap is not None
            ]
            if not candidates:
                estimated_gaps.append(reference_gap)
                continue
            candidate_cluster = self._find_best_gap_cluster(candidates)
            if candidate_cluster is not None:
                estimated_gaps.append(float(candidate_cluster["center"]))
                continue
            nearest_gap = min(candidates, key=lambda gap: abs(gap - reference_gap))
            estimated_gaps.append(float(nearest_gap))
        return estimated_gaps[: self._gap_count]

    def _normalize_gap_to_reference(
        self,
        observed_gap: float,
        reference_gap: float,
    ) -> float | None:
        if observed_gap <= 0.0 or reference_gap <= 0.0:
            return None

        base_radius = max(self._gap_tolerance_px, 0.12 * reference_gap)
        best_gap: float | None = None
        best_err: float | None = None

        for multiple in range(1, self._max_gap_multiple + 1):
            target_gap = float(multiple) * reference_gap
            tolerance = max(base_radius * float(multiple), 0.12 * target_gap)
            err = abs(observed_gap - target_gap)
            if err > tolerance:
                continue
            normalized_gap = observed_gap / float(multiple)
            if best_err is None or err < best_err - 1e-6:
                best_err = err
                best_gap = normalized_gap

        return best_gap

    def _collect_neighbor_gaps(self, lines: list[dict[str, any]]) -> list[float]:
        ordered = sorted(lines, key=lambda line: line["rho"])
        gaps: list[float] = []
        for idx in range(len(ordered) - 1):
            gap = float(ordered[idx + 1]["rho"] - ordered[idx]["rho"])
            if gap > 1.5:
                gaps.append(gap)
        return gaps

    def _find_best_gap_cluster(self, gaps: list[float]) -> dict[str, any] | None:
        ordered = sorted(float(gap) for gap in gaps if gap > 0.0)
        if not ordered:
            return None

        best_cluster: dict[str, any] | None = None
        for start in range(len(ordered)):
            for end in range(start, len(ordered)):
                cluster = ordered[start : end + 1]
                center = sum(cluster) / len(cluster)
                if max(abs(gap - center) for gap in cluster) > self._gap_tolerance_px:
                    continue
                candidate = {
                    "center": center,
                    "count": len(cluster),
                    "err": sum(abs(gap - center) for gap in cluster),
                }
                if best_cluster is None:
                    best_cluster = candidate
                    continue

                if candidate["count"] > best_cluster["count"]:
                    best_cluster = candidate
                    continue
                
                if candidate["count"] == best_cluster["count"] and candidate["err"] < best_cluster["err"] - 1e-6:
                    best_cluster = candidate
                    continue
                
                if (
                    candidate["count"] == best_cluster["count"]
                    and abs(candidate["err"] - best_cluster["err"]) <= 1e-6
                    and candidate["center"] > best_cluster["center"]
                ):
                    best_cluster = candidate
        return best_cluster

    def _update_gaps(self, previous_gaps: list[float], frame_gaps: list[float]) -> list[float]:
        if not previous_gaps:
            return sorted(float(value) for value in frame_gaps)
        
        if not frame_gaps:
            return list(previous_gaps)
        
        updated = sorted(float(value) for value in previous_gaps)
        observed = sorted(float(value) for value in frame_gaps)
        overlap = min(len(updated), len(observed))

        for idx in range(overlap):
            updated[idx] = (1.0 - self._ema) * updated[idx] + self._ema * observed[idx]
        
        if len(updated) < self._gap_count:
            for value in observed[overlap:]:
                updated.append(float(value))
                if len(updated) >= self._gap_count:
                    break
        
        return sorted(updated)
