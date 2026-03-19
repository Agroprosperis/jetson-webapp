
import math
import numpy as np

from scipy.cluster.hierarchy import fcluster, linkage
from .line import Line


class LineClusterizer:
    """Cluster detected lines by 1D axis position into orientation-specific line candidates."""

    def __init__(
        self,
        gap_ratio: float = 0.02,
        gap_min_px: float = 6.0,
        gap_max_px: float = 20.0,
        dedupe_ratio: float = 0.35,
        dedupe_min_px: float = 3.0,
        dedupe_max_px: float = 8.0,
        edge_dedupe_px: float = 4.0,
    ) -> None:
        """Configure the clustering distance.

        Args:
            gap_ratio: Fraction of the relevant frame axis used to derive the base cluster distance.
            gap_min_px: Lower bound for the cluster distance threshold in pixels.
            gap_max_px: Upper bound for the cluster distance threshold in pixels.
        """
        self._gap_ratio = gap_ratio
        self._gap_min_px = gap_min_px
        self._gap_max_px = gap_max_px
        self._dedupe_ratio = dedupe_ratio
        self._dedupe_min_px = dedupe_min_px
        self._dedupe_max_px = dedupe_max_px
        self._edge_dedupe_px = edge_dedupe_px

    def __call__(
        self,
        lines: list[Line],
        frame_shape: tuple[int, int],
    ) -> list[dict[str, any]]:
        frame_h, frame_w = frame_shape
        merged: list[dict[str, any]] = []
        
        for orientation in ("vertical", "horizontal"):
            orientation_clusters = self._build_for_orientation(lines, orientation, frame_w, frame_h)
            for info in orientation_clusters:
                merged.append(
                    {
                        "orientation": orientation,
                        "rho": float(info["rho"]),
                        "theta": float(info["theta"]),
                        "conf": float(info["conf"]),
                    }
                )
        
        return merged

    def _build_for_orientation(
        self,
        lines: list[Line],
        orientation: str,
        frame_w: int,
        frame_h: int,
    ) -> list[dict[str, float]]:
        items = [line for line in lines if line.orientation == orientation]
        if not items:
            return []

        axis_dim = frame_w if orientation == "vertical" else frame_h
        gap_px = self._gap_ratio * axis_dim
        gap_px = min(self._gap_max_px, max(self._gap_min_px, gap_px))

        clusters = self._cluster_lines(items, gap_px)
        clusters = sorted(
            clusters,
            key=lambda cluster: sum(item.axis_pos for item in cluster) / len(cluster),
        )
        cluster_infos = []
        
        for cluster in clusters:
            weight_sum = sum(max(item.score, 1e-6) for item in cluster)
            rho = sum(item.axis_pos * max(item.score, 1e-6) for item in cluster) / weight_sum
            c2 = sum(max(item.score, 1e-6) * math.cos(2.0 * item.theta) for item in cluster)
            s2 = sum(max(item.score, 1e-6) * math.sin(2.0 * item.theta) for item in cluster)
            theta = 0.5 * math.atan2(s2, c2)
            cluster_infos.append({"rho": rho, "theta": theta, "conf": weight_sum})
        dedupe_px = self._dedupe_ratio * gap_px
        dedupe_px = min(self._dedupe_max_px, max(self._dedupe_min_px, dedupe_px))
        cluster_infos = self._collapse_close_candidates(cluster_infos, dedupe_px)
        return self._suppress_edge_duplicates(cluster_infos, orientation, frame_w, frame_h)

    def _cluster_lines(self, lines: list[Line], distance_threshold: float) -> list[list[Line]]:
        if len(lines) == 1:
            return [list(lines)]
        
        observations = np.array([[line.axis_pos] for line in lines], dtype=np.float64)
        linkage_matrix = linkage(observations, method="complete", metric="euclidean")
        labels = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")
        grouped: dict[int, list[Line]] = {}
        
        for label, line in zip(labels, lines):
            grouped.setdefault(int(label), []).append(line)
        
        return list(grouped.values())

    def _collapse_close_candidates(
        self,
        cluster_infos: list[dict[str, float]],
        distance_threshold: float,
    ) -> list[dict[str, float]]:
        if len(cluster_infos) <= 1:
            return list(cluster_infos)

        ordered = sorted(cluster_infos, key=lambda item: float(item["rho"]))
        collapsed: list[dict[str, float]] = []
        group: list[dict[str, float]] = [ordered[0]]

        for item in ordered[1:]:
            prev = group[-1]
            if float(item["rho"]) - float(prev["rho"]) <= distance_threshold:
                group.append(item)
                continue
            collapsed.append(self._merge_candidate_group(group))
            group = [item]

        collapsed.append(self._merge_candidate_group(group))
        return collapsed

    def _merge_candidate_group(self, group: list[dict[str, float]]) -> dict[str, float]:
        if len(group) == 1:
            return dict(group[0])

        best = max(group, key=lambda item: float(item["conf"]))
        weight_sum = sum(max(float(item["conf"]), 1e-6) for item in group)
        rho = sum(float(item["rho"]) * max(float(item["conf"]), 1e-6) for item in group) / weight_sum
        conf = sum(float(item["conf"]) for item in group)
        return {
            "rho": float(rho),
            "theta": float(best["theta"]),
            "conf": float(conf),
        }

    def _suppress_edge_duplicates(
        self,
        cluster_infos: list[dict[str, float]],
        orientation: str,
        frame_w: int,
        frame_h: int,
    ) -> list[dict[str, float]]:
        if len(cluster_infos) <= 1 or self._edge_dedupe_px <= 0.0:
            return list(cluster_infos)

        kept: list[dict[str, float]] = []
        for candidate in sorted(cluster_infos, key=lambda item: float(item["conf"]), reverse=True):
            if any(
                self._edge_duplicate_distance(candidate, existing, orientation, frame_w, frame_h)
                <= self._edge_dedupe_px
                for existing in kept
            ):
                continue
            kept.append(dict(candidate))

        return sorted(kept, key=lambda item: float(item["rho"]))

    def _edge_duplicate_distance(
        self,
        candidate_a: dict[str, float],
        candidate_b: dict[str, float],
        orientation: str,
        frame_w: int,
        frame_h: int,
    ) -> float:
        a0, a1 = self._edge_intercepts(candidate_a, orientation, frame_w, frame_h)
        b0, b1 = self._edge_intercepts(candidate_b, orientation, frame_w, frame_h)
        return min(abs(a0 - b0), abs(a1 - b1))

    def _edge_intercepts(
        self,
        candidate: dict[str, float],
        orientation: str,
        frame_w: int,
        frame_h: int,
    ) -> tuple[float, float]:
        rho = float(candidate["rho"])
        theta = float(candidate["theta"])
        dx = math.cos(theta)
        dy = math.sin(theta)
        eps = 1e-6

        if orientation == "vertical":
            y_center = 0.5 * float(frame_h)
            if abs(dy) <= eps:
                return rho, rho
            x_top = rho + ((0.0 - y_center) * dx / dy)
            x_bottom = rho + (((float(frame_h) - 1.0) - y_center) * dx / dy)
            return x_top, x_bottom

        x_center = 0.5 * float(frame_w)
        if abs(dx) <= eps:
            return rho, rho
        y_left = rho + ((0.0 - x_center) * dy / dx)
        y_right = rho + (((float(frame_w) - 1.0) - x_center) * dy / dx)
        return y_left, y_right
