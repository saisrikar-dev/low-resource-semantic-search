from __future__ import annotations

import math
from typing import Iterable


def dot_product(left: Iterable[float], right: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(left, right, strict=True))


def l2_norm(values: Iterable[float]) -> float:
    return math.sqrt(sum(value * value for value in values))


def l2_normalize(values: list[float]) -> list[float]:
    norm = l2_norm(values)
    if norm == 0.0:
        return values
    return [value / norm for value in values]

