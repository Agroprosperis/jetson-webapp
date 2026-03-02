import math
from dataclasses import dataclass

from .line import Line


@dataclass(slots=True)
class _AccumulatedFrame:
    lines: list[Line]
    dx: float = 0.0
    dy: float = 0.0


class TemporalLineAccumulator:
    def __init__(
        self,
        history_size: int = 5,
        keep_top_ratio: float = 1.0,
        min_keep_per_orientation: int = 4,
        score_decay: float = 0.9,
    ) -> None:
        self._history_size = max(int(history_size), 1)
        self._keep_top_ratio = min(max(float(keep_top_ratio), 0.0), 1.0)
        self._min_keep_per_orientation = max(int(min_keep_per_orientation), 1)
        self._score_decay = min(max(float(score_decay), 0.0), 1.0)
        self._history: list[_AccumulatedFrame] = []

    def __call__(
        self,
        lines: list[Line],
        flow_shift: dict[str, float] | None = None,
    ) -> list[Line]:
        flow_data = flow_shift or {"dx": 0.0, "dy": 0.0}
        self._advance_offsets(
            float(flow_data.get("dx", 0.0)),
            float(flow_data.get("dy", 0.0)),
        )
        selected_lines = self._select_lines(lines)
        if selected_lines:
            self._history.insert(0, _AccumulatedFrame(lines=selected_lines))
            if len(self._history) > self._history_size:
                self._history = self._history[: self._history_size]
        return self._materialize_lines()

    def _advance_offsets(self, dx: float, dy: float) -> None:
        for entry in self._history:
            entry.dx += dx
            entry.dy += dy

    def _select_lines(self, lines: list[Line]) -> list[Line]:
        selected: list[Line] = []
        for orientation in ("vertical", "horizontal"):
            orientation_lines = [line for line in lines if line.orientation == orientation]
            if not orientation_lines:
                continue
            keep_count = max(
                self._min_keep_per_orientation,
                int(math.ceil(len(orientation_lines) * self._keep_top_ratio)),
            )
            keep_count = min(len(orientation_lines), keep_count)
            selected.extend(
                sorted(orientation_lines, key=lambda item: item.score, reverse=True)[:keep_count]
            )
        return selected

    def _materialize_lines(self) -> list[Line]:
        if not self._history:
            return []

        frame_weights = [self._score_decay ** idx for idx in range(len(self._history))]
        weight_sum = sum(frame_weights)
        if weight_sum <= 0.0:
            frame_weights = [1.0] + [0.0] * (len(self._history) - 1)
            weight_sum = 1.0

        accumulated: list[Line] = []
        for idx, entry in enumerate(self._history):
            frame_weight = frame_weights[idx] / weight_sum
            for line in entry.lines:
                axis_shift = entry.dx if line.orientation == "vertical" else entry.dy
                accumulated.append(
                    Line(
                        x1=line.x1 + entry.dx,
                        y1=line.y1 + entry.dy,
                        x2=line.x2 + entry.dx,
                        y2=line.y2 + entry.dy,
                        score=float(line.score) * frame_weight,
                        orientation=line.orientation,
                        axis_pos=line.axis_pos + axis_shift,
                        theta=line.theta,
                    )
                )
        return accumulated
