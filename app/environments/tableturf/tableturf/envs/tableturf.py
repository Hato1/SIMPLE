from typing import List, Tuple, Optional

import gym
import numpy as np

from deck import Deck

# import config

if __name__ == "__main__":
    import logging
    logger = logging.getLogger()
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
else:
    from stable_baselines import logger

from board import Board
from helpers import create_universal_deck, Move, Point
from player import Player


universal_deck = create_universal_deck()
PASS = 19 * 25 * 175 * 2


class TableturfEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, verbose=False, manual=False):
        super(TableturfEnv, self).__init__()

        self.name = 'tableturf'
        self.manual = manual

        self.n_players = 2
        self.card_types = 175
        self.n_turns = 12

        #
        # Observation + Action spaces
        # ---------------------------
        #  Observations:
        #       19x25 board with 5 states for each tile +
        #       4/175 cards in hand +
        #       0-15/175 cards in deck +
        #       X special tile charges +
        #       X moves left.
        #
        #  Actions:
        #       (every board position) x (4 cards in hand) x (Special, pass or neither)
        #
        # Passing also requires discard
        # Note: not every action is legal
        #
        # ToDo: Complete observation space.
        self.observation_space = gym.spaces.Box(np.float32(-2), np.float32(2), (19, 25))
        self.action_space = gym.spaces.Discrete(19 * 25 * 175 * 2 + 175)

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
        self.current_player_num = 0

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
        # ToDo: Implement observation
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
        # for p in self.players:  # TODO this should be from reference point of the current_player
        #     ret = np.append(ret, p.score / self.max_score)
        #
        # ret = np.append(ret, self.legal_actions)
        #
        # return ret

    def action_to_move(self, action: int) -> Optional[Move]:
        assert self.action_space.contains(action)
        if action == PASS:
            return None
        x, action = action % 19, action // 19
        y, action = action % 25, action // 25
        c, action = action % 175, action // 175
        s, action = action % 2, action // 2
        return Move(universal_deck[c], Point(x, y), bool(s))

    def move_to_action(self, move: Move) -> int:
        if move:
            action = 0
            action += move.point.x
            action += move.point.y * 19
            action += (move.card.id - 1) * 19 * 25
            action += move.special * 19 * 25 * 175
        else:
            action = PASS
        assert self.action_space.contains(action)
        return action

    def check_legal_action(self, action: int) -> bool:
        move = self.action_to_move(action)
        if not move:
            return True
        if move.card not in self.current_player.hand:
            return False
        if move.special and self.current_player.special_charges < move.card.cost:
            return False
        if not self.board.check_legal_action(move):
            return False
        return True

    def play_card(self, other_action: int, my_action: int):
        my_move = self.action_to_move(my_action)
        other_move = self.action_to_move(other_action)

        for p, move in zip(Player.players, [my_move, other_move]):
            if move:
                p.play(move.card)
                logger.debug(f"Player {p.id} playing {move.card.name}")
            else:
                p.special_charges += 1
                logger.debug(f"Player {p.id} passes")

        self.board.play(my_move, other_move)

    def score_game(self) -> List[float]:
        """Determine winner and return rewards."""
        p1, p2 = self.board.score()
        if p1 > p2:
            return [1.0, -1.0]
        elif p1 < p2:
            return [-1.0, 1.0]
        else:
            logger.error("Players tied! Not implemented.")
            return [0.0, 0.0]

    def step(self, action: int) -> Tuple[np.ndarray, List[float], bool, dict]:
        """Take a turn.

        Returns:
            Observation: An element of the observation space
            Reward: Reward for taking this action
            Terminal: Reached a terminal game state.
            Truncated: End the game prematurely before a terminal state is reached.
            Info: Debug comments.

        # ToDo: Reward for capturing tiles?
        """
        reward = [0] * self.n_players
        done = False

        # check move legality
        if not self.check_legal_action(action):
            reward = [1.0 / (self.n_players - 1)] * self.n_players
            reward[self.current_player_num] = -1
            done = True

        # play the card
        else:
            self.action_bank.append(action)

            # If both players have made an action, carry it out.
            if len(self.action_bank) == self.n_players:
                logger.debug(f'The chosen cards are now played simultaneously')
                self.play_card(*self.action_bank)
                self.action_bank = []

            self.current_player_num = (self.current_player_num + 1) % self.n_players

            if self.current_player_num == 0:
                self.turns_taken += 1

            if self.turns_taken == self.n_turns:
                reward = self.score_game()
                done = True

            self.render()

        self.done = done
        # (observation, reward, terminated/truncated, info)
        return self.observation, reward, done, {}

    def render(self, mode='human', close=False):
        # ToDo: Display Board state.
        if close:
            return
        logger.debug(f'\n\n-------TURN {self.turns_taken + 1}-----------')
        logger.debug(f"It is Player {self.current_player.id}'s turn to choose\n")
        for p in Player.players:
            logger.debug(p)
        if self.verbose:
            pass
            # ToDo: Log observation
            # logger.debug(
            #     f'\nObservation: \n{[i if o == 1 else (i, o) for i, o in enumerate(self.observation) if o != 0]}')
        if self.done:
            logger.debug(f'\n\nGAME OVER')
        logger.debug(f'Scores {self.board.score()}')

    def rules_move(self):
        raise Exception('Rules based agent is not yet implemented for Tableturf battle!')

    def seed(self, seed=0):
        pass


def test():
    env = TableturfEnv(verbose=True, manual=True)
    env.reset()

    env.render()


def test_action_to_move():
    env = TableturfEnv(verbose=True, manual=True)
    env.reset()
    for i in range(env.action_space.n):
        move = env.action_to_move(i)
        action = env.move_to_action(move)
        assert i == action


if __name__ == "__main__":
    test()
