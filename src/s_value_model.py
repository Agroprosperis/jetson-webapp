from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SValueModel:
    numerator: float = 1111.0
    denominator: float = 400.0
    decimals: int = 1

    def calculate(self, object_count: int | float) -> float:
        value = (float(object_count) * self.numerator) / self.denominator
        return round(value, self.decimals)


DEFAULT_S_VALUE_MODEL = SValueModel()


def calculate_s_value(object_count: int | float) -> float:
    return DEFAULT_S_VALUE_MODEL.calculate(object_count)
