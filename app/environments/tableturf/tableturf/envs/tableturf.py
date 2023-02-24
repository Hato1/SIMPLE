from typing import List, Tuple, Union

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    import gym

import numpy as np
from typing_extensions import Literal

from colour import C
from deck import Deck

# import config

import logging
# Importing stable_baselines takes forever, use builtin for quicker testing.
# from stable_baselines import logger

from board import Board, IllegalMoveError
from helpers import create_universal_deck, Move, Point, Pass
from player import Player

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG, format='%(message)s')


universal_deck = create_universal_deck()
PASS = 19 * 26 * 175 * 2


class TableturfEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose=False, manual=False):
        super(TableturfEnv, self).__init__()

        self.name = 'tableturf'
        self.manual = manual

        self.n_players = 2
        self.card_types = 175
        self.width = 19
        self.height = 26
        self.n_turns = 12

        #
        # Observation + Action spaces
        # ---------------------------
        #  Observations:
        #       19x26 board with 5 states for each tile +
        #       4/175 cards in hand +
        #       0-15/175 cards in deck +
        #       X special tile charges +
        #       X moves left.
        #
        #  Actions:
        #       (every board position) x (175 cards to play from hand) x (Special or not)
        #       + 175 cards to pass with
        #
        # Passing also requires discard
        # Note: not every action is legal
        #
        # FixMe: Implement observation space
        self.observation_space = gym.spaces.Box(np.float32(-2), np.float32(2), (self.width, self.height))
        self.action_space = gym.spaces.Discrete(self.width * self.height * self.card_types * 2 + self.card_types)

        self.verbose = verbose

    def reset(self) -> np.ndarray:
        """Reset game state and get observation."""
        self.turns_taken = 0
        self.board = Board()
        self.action_bank = []

        # Initialise both players with a random deck of 15 cards.
        Player.reset()
        Player(Deck(universal_deck.get(15)))
        Player(Deck(universal_deck.get(15)))
        self.current_player_num: Literal[0, 1] = 0

        self.done = False
        logger.debug(f'\n\n---- NEW GAME ----')
        return self.observation

    @property
    def current_player(self) -> Player:
        return Player.get(self.current_player_num)

    @property
    def current_opponent(self) -> Player:
        return Player.get(1 - self.current_player_num)

    @property
    def observation(self):
        # FixMe: Implement observation space
        logger.critical("Observation is not implemented")
        return np.ndarray([])
        # obs = np.zeros(([self.total_positions, self.card_types]))
        # player_num = self.current_player_num
        # hands_seen = 0
        #
        # for i in range(self.n_players):
        #     player = self.players[player_num]
        #
        #     if self.turns_taken >= hands_seen:
        #         for card in player.hand.cards:
        #             obs[i * 2][card.id] = 1
        #
        #     for card in player.position.cards:
        #         obs[i * 2 + 1][card.id] = 1
        #
        #     player_num = (player_num + 1) % self.n_players
        #     hands_seen += 1
        #
        # if self.turns_taken >= self.n_players - 1:
        #     for card in self.deck.cards:
        #         obs[6][card.id] = 1
        #
        # for card in self.discard.cards:
        #     obs[7][card.id] = 1
        #
        # ret = obs.flatten()
        # for p in self.players:  # to do this should be from reference point of the current_player
        #     ret = np.append(ret, p.score / self.max_score)
        #
        # ret = np.append(ret, self.legal_actions)
        #
        # return ret

    def action_to_move(self, action: int) -> Union[Move, Pass]:
        assert self.action_space.contains(action)
        if action >= PASS:
            c = action - PASS
            return Pass(universal_deck[c])
        x, action = action % self.width, action // self.width
        y, action = action % self.height, action // self.height
        c, action = action % self.card_types, action // self.card_types
        s, action = action % 2, action // 2
        return Move(universal_deck[c], Point(x, y), bool(s), self.current_player_num)

    def move_to_action(self, move: Union[Move, Pass]) -> int:
        if isinstance(move, Move):
            action = 0
            action += move.point.x
            action += move.point.y * self.width
            action += (move.card.id - 1) * self.width * self.height
            action += move.special * self.width * self.height * self.card_types
        else:
            action = PASS + move.card.id - 1
        assert self.action_space.contains(action)
        return action

    def check_legal_action(self, move: Union[Move, Pass]) -> bool:
        if move.card not in self.current_player.hand:
            logger.debug(f"{repr(move.card)} not in {self.current_player}'s hand.")
            return False
        if isinstance(move, Pass):
            return True
        if move.special and self.current_player.special_charges < move.card.cost:
            logger.debug(
                f"{self.current_player} cannot afford special ({self.current_player.special_charges < move.card.cost})."
            )
            return False
        try:
            self.board.check_legal_action(move)
        except IllegalMoveError as e:
            logger.debug(e)
            return False
        return True

    def play_card(self, p1_move: Union[Move, Pass], p2_move: Union[Move, Pass]):
        moves: List[Move] = []
        for p, move in zip(Player.players, [p1_move, p2_move]):
            logger.debug(f"{p} chose {repr(move.card)}")
            p.play(move.card)
            if isinstance(move, Pass):
                logger.debug(f"Player {p.id} passes")
                p.special_charges += 1
            else:
                moves.append(move)
                if move.special:
                    p.special_charges -= move.card.cost
                    assert p.special_charges >= 0

        self.board.play(moves)

    def score_game(self) -> List[float]:
        """Determine winner and return rewards."""
        p0, p1 = self.board.score()
        if p0 > p1:
            logger.debug(f"{Player.players[0]} wins!")
            return [1.0, -1.0]
        elif p0 < p1:
            logger.debug(f"{Player.players[1]} wins!")
            return [-1.0, 1.0]
        else:
            logger.error("Players tied! Not implemented, giving reward of 0.")
            return [0.0, 0.0]

    def step(self, action: int) -> Tuple[np.ndarray, List[float], bool, dict]:
        """Take a turn.

        Returns:
            Observation: An element of the observation space
            Reward: Reward for taking this action
            Terminal: Reached a terminal game state.
            Truncated: End the game prematurely before a terminal state is reached.
            Info: Debug comments.

        # ToDo: Should I add a reward for capturing tiles to assist model learning?
        """
        reward = [0] * self.n_players
        done = False
        move = self.action_to_move(action)

        # check move legality
        if not self.check_legal_action(move):
            reward = [1.0 / (self.n_players - 1)] * self.n_players
            reward[self.current_player_num] = -1
            done = True

        # play the card
        else:
            self.action_bank.append(move)

            # If both players have made an action, carry it out.
            if len(self.action_bank) == self.n_players:
                logger.debug(f'The chosen cards are now played simultaneously')
                self.play_card(*self.action_bank)
                self.action_bank = []

            self.current_player_num = (self.current_player_num + 1) % self.n_players

            if self.current_player_num == 0:
                self.turns_taken += 1

            self.render()

            if self.turns_taken == self.n_turns:
                reward = self.score_game()
                done = True

        self.done = done
        return self.observation, reward, done, {}

    def render(self, mode='human', close=False):
        if close:
            return
        logger.debug(f'\n\n-------TURN {self.turns_taken + 1}-----------')
        logger.debug(f"It is {self.current_player}'s turn to choose\n")
        for p in Player.players:
            logger.debug(repr(p))
        if self.verbose:
            pass
            # FixMe: Implement observation space
            # logger.debug(
            #     f'\nObservation: \n{[i if o == 1 else (i, o) for i, o in enumerate(self.observation) if o != 0]}')
        logger.debug(self.board)
        if self.done:
            logger.debug(f'\n\nGAME OVER')
        scores = self.board.score()
        logger.debug(f'Scores {Player.players[0].c}{scores[0]}{C.END}, {Player.players[1].c}{scores[1]}{C.END}\n')

    def rules_move(self):
        raise NotImplementedError('Rules based agent is not yet implemented for Tableturf battle!')

    def seed(self, seed=0):
        pass
