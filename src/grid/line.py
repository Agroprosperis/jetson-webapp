from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Line:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    orientation: str
    axis_pos: float
    theta: float

