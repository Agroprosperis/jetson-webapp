
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
        return cluster_infos

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
