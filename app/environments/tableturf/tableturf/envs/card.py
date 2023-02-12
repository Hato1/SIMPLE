"""
There are 175 unique cards. Each card has a title, a value, is at most 7x6
tiles, represented by a numpy array and optionally one special space
represented as an index into the numpy array.
"""
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np


@dataclass
class Card:
    # Todo: How many special charges does it cost to play?
    id: int
    name: str
    priority: int
    cost: int
    shape: np.ndarray
    special: Tuple[int, int] = None

    def __repr__(self):
        return f"Card {self.id}: {self.name}"

    def __str__(self):
        # Todo: Render special tile, if it exists
        msg = f"Card {self.id}: {self.name}\n"
        shape: List = self.shape.tolist()  # type: ignore
        for row in shape:
            row = ['▣' if x else ' ' for x in row]
            msg += ''.join(row) + '\n'
        return msg
