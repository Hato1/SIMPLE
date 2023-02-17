"""
There are 175 unique cards. Each card has a title, a value, is at most 7x6
tiles, represented by a numpy array and optionally one special space
represented as an index into the numpy array.
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from colour import C


@dataclass
class Point:
    x: int
    y: int


@dataclass
class Card:
    id: int
    name: str
    priority: int
    cost: int
    splat_zone: np.ndarray
    special: Optional[Point] = None

    def __repr__(self):
        return f"{C.PURPLE}Card {self.id}:{C.BLUE} {self.name}{C.END}"

    def __str__(self):
        msg = f"Card {self.id}: {self.name}\n"
        splat_zone: List = self.splat_zone.tolist()  # type: ignore
        for i, row in enumerate(splat_zone):
            row = ['▣' if x else ' ' for x in row]
            if self.special and self.special.x == i:
                row[self.special.y] = "\033[94m▣\033[0m"
            msg += ''.join(row) + '\n'
        return msg
