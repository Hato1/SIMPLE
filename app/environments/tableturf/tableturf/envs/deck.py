"""
A deck has 15 cards. Because a game is 12 turns long, 3 of these cards won't
be played.
"""
import random
from typing import Optional, List

from card import Card


class Deck:
    def __init__(self, cards: Optional[List[Card]] = None):
        self.cards = cards if cards else []

    def shuffle(self):
        random.shuffle(self.cards)

    def sort(self):
        self.cards.sort(key=lambda x: x.id)

    def draw(self, n: int):
        drawn = []
        for x in range(n):
            if not self.cards:
                return drawn
            drawn.append(self.cards.pop())
        return drawn

    def get(self, n: int):
        return random.sample(self.cards, n)

    def __repr__(self):
        output = ""
        for card in self.cards:
            output += str(card) + "\n"
        return output

    def __getitem__(self, item):
        for card in self.cards:
            if card.id == item + 1:
                return card
        raise ValueError(f"Card '{item + 1}' not in deck.")
