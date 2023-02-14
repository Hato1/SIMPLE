from __future__ import annotations

import random
from typing import Dict, Optional, Tuple
from typing_extensions import Literal

import numpy as np

from colour import C
from helpers import Move, Pass

import logging
logger = logging.getLogger()


class Board:
    templates: Dict[str, Board] = {}

    def __init__(self, board: Optional[np.ndarray] = None, template_name: Optional[str] = None):
        if board is not None:
            self.board = board
        else:
            self.board = random.choice(list(self.templates.values())).board.copy()
        if template_name:
            self.templates[template_name] = self

    def score(self) -> Tuple[int, int]:
        p0 = np.count_nonzero(self.board > 0)
        p1 = np.count_nonzero(self.board < 0)
        return p0, p1

    def _play(self, move: Move, player_id: Literal[0, 1]):
        # Todo
        pass

    def play_conflict(self, p0, p1):
        # Todo
        pass

    def play(self, p0: Optional[Move], p1: Optional[Move]):
        p0_pass, p1_pass = isinstance(p0, Pass), isinstance(p1, Pass)
        if p0_pass and p1_pass:
            pass
        elif p0_pass:
            self._play(p1, 1)
        elif p1_pass:
            self._play(p0, 0)
        else:
            assert self.check_legal_action(p0), f"Card {p0.card} cannot be played at {p0.point}. Special: {p0.special}."
            assert self.check_legal_action(p1), f"Card {p1.card} cannot be played at {p1.point}. Special: {p1.special}."
            if p0.card.priority > p1.card.priority:
                self._play(p0, 0)
                self._play(p1, 1)
            elif p0.card.priority < p1.card.priority:
                self._play(p1, 1)
                self._play(p0, 0)
            else:
                self.play_conflict(p0, p1)

    def check_legal_action(self, move: Move):
        # ToDo: Check legal action.
        logger.error("Check legal action not implemented, returning True.")
        return True

    @property
    def height(self):
        return self.board.shape[0]

    @property
    def width(self):
        return self.board.shape[1]

    def __repr__(self):
        symbols = {
            -2: f"{C.MAGENTA}■{C.END}",
            -1: f"{C.CYAN}■{C.END}",
            0: f" ",
            1: f"{C.YELLOW}◼{C.END}",
            2: f"{C.RED}◼{C.END}"
        }
        msg = "┌" + "─" * self.width + "┐\n"
        for row in self.board:
            msg += "│"
            for col in row:
                msg += symbols[col.item()]
            msg += "│\n"

        msg += "└" + "─" * self.width + "┘"
        return msg


# 9 x 26
main_street = np.zeros((26, 9), dtype=int)
main_street[3, 4] = -2
main_street[-4, 4] = 2
Board(main_street, template_name="Main Street")
