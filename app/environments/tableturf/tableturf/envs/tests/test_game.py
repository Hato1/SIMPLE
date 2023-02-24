from card import Point
from helpers import Move
from tableturf import TableturfEnv

env = TableturfEnv(verbose=True, manual=True)


def always_play_first_card():
    while not env.done:
        play_first_card()


def play_first_card():
    point = Point(3, 21) if not env.current_player.id else Point(3, 4)
    move = Move(env.current_player.hand[0], point, False, env.current_player_num)
    action = env.move_to_action(move)
    env.step(action)


def test_play_card():
    env.reset()
    env.render()
    play_first_card()
    play_first_card()


def test_action_to_move():
    env.reset()
    for i in range(env.action_space.n):
        move = env.action_to_move(i)
        action = env.move_to_action(move)
        assert i == action
