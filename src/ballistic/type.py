from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from ballistic import util


@dataclass(frozen=True, eq=True)
class Top: pass

@dataclass(frozen=True, eq=True)
class Dist:
    mean : Range | None 
    devi : Range | None
    total : Range | None

@dataclass(frozen=True, eq=True)
class Num:
    range : Range | None 

@dataclass(frozen=True, eq=True)
class Range:
    lower: float
    upper: float

Ty = Top | Dist | Num