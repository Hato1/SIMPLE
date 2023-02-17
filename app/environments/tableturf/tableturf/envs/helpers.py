import logging
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Tuple, Optional

import numpy as np
from typing_extensions import Literal

from card import Card
from deck import Deck

card_path = Path("environments/tableturf/tableturf/envs/cards")


@dataclass
class Point:
    x: int
    y: int


class Move(NamedTuple):
    """Python representation of a non-passing action."""
    card: Card
    point: Point
    special: bool
    player_id: Literal[0, 1]
    # FixMe: All 4 rotations
    rotation: int = 0


class Pass(NamedTuple):
    """Python representation of a passing action."""
    card: Card


def read_shape(file: Path) -> Tuple[np.ndarray, Optional[Point]]:
    with open(file) as f:
        lines = f.readlines()
    lines = [list(line[:-1]) for line in lines]

    special = None
    for i, line in enumerate(lines):
        if "S" in line:
            special = Point(i, line.index("S"))

    # Convert to bools
    lines = [[x != ' ' for x in y] for y in lines]

    # Make all lists equal length
    columns = max([len(x) for x in lines])
    for line in lines:
        while len(line) < columns:
            line.append(False)

    # Check card file is of minimum dimensions
    try:
        assert any([x for x in lines[0]])
        assert any([x[0] for x in lines])
        assert any([x[-1] for x in lines])
        assert any([x for x in lines[-1]])
    except AssertionError:
        logging.critical(f"Invalid card file: '{file}'")

    return np.array(lines), special


def create_universal_deck(path: Path = card_path) -> Deck:
    """Reads card files from path and adds them all to a deck."""
    cards = []
    ids = []
    for card_file in sorted(path.iterdir()):
        if card_file.name.count(",") != 3:
            continue
        id_, name, priority, cost = card_file.name.split(",")

        splat_zone, special = read_shape(card_file)

        ids.append(int(id_))
        cards.append(Card(int(id_), name, int(priority), int(cost), splat_zone, special))

    # Pad deck
    for i in range(1, 176):
        if i not in ids:
            cards.append(Card(i, "Dummy", 0, 0, np.array([[True]])))
    deck = Deck(cards)
    deck.sort()
    return deck
