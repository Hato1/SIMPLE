from __future__ import annotations

import random
from typing import Dict, Optional, Tuple, List

from scipy.signal import convolve2d

import numpy as np

from colour import C
from helpers import Move

import logging
logger = logging.getLogger()


class IllegalMoveError(Exception):
    pass


class Board:
    """Board states. Always index by Y first, then X."""
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

    def _play(self, move: Move):
        # Todo play card
        try:
            self.check_legal_action(move)
        except IllegalMoveError:
            logger.debug(f"{repr(move.card)} cannot be played at {move.point}. Special: {move.special}.")
            raise

        # get mask
        # apply mask
        # add special tile

    def _play_conflict(self, moves: List[Move]):
        # Fixme: Play conflict
        # Special Spaces can only be covered by other Special Spaces.
        # Special Spaces do not become walls unless colliding with another special space.

        # Create both masks.
        # Get intersection of masks, as another mask.
        # Apply both masks.
        # Apply intersection mask as stone.
        # Add both specials, provided they aren't at the same coordinate.
        pass

    def play(self, moves: List[Move]):
        if moves[0].card.priority == moves[1].card.priority:
            self._play_conflict(moves)
        else:
            moves.sort(key=lambda x: x.card.priority)
            for move in moves:
                logger.debug(f"{repr(move.card)} goes first with priority of {move.card.priority}.")
                self._play(move)

    def assert_fits_on_board(self, x_min: int, x_max: int, y_min: int, y_max: int):
        """Checks dimensions lie within the board."""
        if any([x_min, y_min]) < 0 or x_max > self.board.shape[1] or y_max > self.board.shape[0]:
            raise IllegalMoveError("Shape doesn't fit on board!")

    def assert_space_on_board(self, mask: np.ndarray, special: bool, verbose: bool):
        """Check the mask doesn't overlap any illegal tiles on the board.

        0 values can always be overlapped.
        1 values can be overlapped by special moves.
        2 values can never be overlapped.
        FixMe: Define Stone. Stone can never be overlapped.
        """
        if not all(np.abs(self.board[mask]) <= (1 if special else 0)):
            if verbose:
                debug_board = self.board.copy()
                debug_board[mask] = 3
                real_board, self.board = self.board, debug_board
                logger.debug(self)
                self.board = real_board
            raise IllegalMoveError("Placement overlaps illegal tiles!")

    def assert_adjacency(self, mask: np.ndarray, player_id: int, special: bool, verbose: bool):
        """Check that at-least one tile of the mask is adjacent to the players tile.

        If tile is special, it must be adjacent to a special players tile.
        """
        kernel = np.asarray([
            [True, True, True],
            [True, True, True],
            [True, True, True]]
        )
        adjacencies = convolve2d(mask.astype(int), kernel.astype(int), mode="same").astype(bool)

        mini = 1 if special else 0
        if player_id == 0 and any(self.board[adjacencies] > mini):
            return True
        elif player_id == 1 and any(self.board[adjacencies] < -mini):
            return True
        else:
            if verbose:
                debug_board = self.board.copy()
                debug_board[adjacencies] = -3 if player_id else 3
                debug_board[mask] = -4 if player_id else 4
                real_board, self.board = self.board, debug_board
                logger.debug(self)
                self.board = real_board
            raise IllegalMoveError("Placement isn't adjacent to player")

    def check_legal_action(self, move: Move, verbose=True):
        """Check a Move is legal.

        A move is legal if it doesn't overlap any tiles (Regular tiles can be overlapped with special).
        It must also be adjacent to a friendly coloured tile (Must be special tile if special).
        """

        splat_zone = np.rot90(move.card.splat_zone, move.rotation)
        x_min, y_min = move.point.x, move.point.y
        x_max, y_max = x_min + splat_zone.shape[1], y_min + splat_zone.shape[0]

        self.assert_fits_on_board(x_min, x_max, y_min, y_max)

        mask = np.zeros(self.board.shape, dtype=bool)
        mask[y_min:y_max, x_min:x_max] = splat_zone

        self.assert_space_on_board(mask, move.special, verbose)
        self.assert_adjacency(mask, move.player_id, move.special, verbose)

    @property
    def height(self):
        return self.board.shape[0]

    @property
    def width(self):
        return self.board.shape[1]

    def __repr__(self):
        symbols = {
            -4: f"{C.PURPLE}x{C.END}",
            -3: f"{C.CYAN}x{C.END}",
            -2: f"{C.PURPLE}■{C.END}",
            -1: f"{C.CYAN}■{C.END}",
            0: f" ",
            1: f"{C.YELLOW}◼{C.END}",
            2: f"{C.RED}◼{C.END}",
            3: f"{C.YELLOW}x{C.END}",
            4: f"{C.RED}x{C.END}"
        }
        msg = "┌" + "─" * self.width + "┐\n"
        for row in self.board:
            msg += "│"
            for col in row:
                msg += symbols[col.item()]
            msg += "│\n"

        msg += "└" + "─" * self.width + "┘"
        return msg


main_street = np.zeros((26, 9), dtype=int)
main_street[3, 4] = -2
main_street[-4, 4] = 2
Board(main_street, template_name="Main Street")
