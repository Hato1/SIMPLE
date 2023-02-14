from __future__ import annotations

from typing import List

from card import Card
from colour import C
from deck import Deck


class Player:
    hand_size = 4
    players: List[Player] = []

    def __init__(self, deck: Deck):
        self.deck = deck
        self.hand: List[Card] = []
        self.draw()
        self.special_charges = 0
        self.id = len(self.players)
        self.players.append(self)

    def draw(self, n=hand_size):
        """Draw cards from deck until we have n cards in hand."""
        cards_to_draw = n - len(self.hand)
        self.hand.extend(self.deck.draw(cards_to_draw))

    def play(self, card: Card):
        self.hand.remove(card)
        self.draw()

    def pick(self, name):
        for i, c in enumerate(self.hand):
            if c.name == name:
                self.hand.pop(i)
                return c

    @property
    def c(self):
        return C.YELLOW if self.id == 0 else C.CYAN

    @classmethod
    def reset(cls):
        cls.players = []

    @classmethod
    def get(cls, player: int):
        return cls.players[player]

    def __str__(self):
        return f"{self.c}Player {self.id}{C.END}"

    def __repr__(self):
        output = f"{self.c}Player {self.id}{C.END}'s hand: "
        for card in self.hand:
            output += repr(card) + ", "
        output = output[:-2]
        output += f". {C.RED}Charges: {C.DARKRED}{self.special_charges}{C.END}."
        return output
