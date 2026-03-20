from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SValueModel:
    numerator: float = 1111.0
    denominator: float = 400.0
    decimals: int = 1

    def calculate(self, total_unique_objects: int | float) -> float:
        value = (float(total_unique_objects) * self.numerator) / self.denominator
        return round(value, self.decimals)


DEFAULT_S_VALUE_MODEL = SValueModel()


def calculate_s_value(total_unique_objects: int | float) -> float:
    return DEFAULT_S_VALUE_MODEL.calculate(total_unique_objects)
