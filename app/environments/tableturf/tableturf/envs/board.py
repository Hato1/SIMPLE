from __future__ import annotations

import random
from typing import Dict, Optional, Tuple

import numpy
import numpy as np

from helpers import Move

import logging
logger = logging.getLogger()


class Board:
    base_boards: Dict[str, Board] = {}

    def __init__(self, board: Optional[np.ndarray] = None, base: Optional[str] = None):
        self.board = board or random.choice(list(self.base_boards.values())).board.copy()
        if base:
            self.base_boards[base] = self

    def score(self) -> Tuple[float, float]:
        logger.error("Board scoring is not implemented.")
        return 0.0, 0.0

    def play(self, p1: Optional[Move], p2: Optional[Move]):
        raise NotImplementedError

    def check_legal_action(self, move: Move):
        raise NotImplementedError


Board(numpy.ndarray([]), base="Dummy")
