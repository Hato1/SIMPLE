# docker-compose exec app python3 test.py -d -g 1 -a base base human -e butterfly 

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import random
import argparse
from statistics import mean

from stable_baselines import logger
from stable_baselines.common import set_global_seeds

from utils.files import load_all_models_with_names, write_results, write_tournament_results
from utils.register import get_environment
from utils.agents import Agent

import config

class PlayerScore():
    def __init__(self, name):
        self.name = name
        self.scores = []

    

def main(args):

    logger.configure(config.LOGDIR)

    if args.debug:
        logger.set_level(config.DEBUG)
    else:
        logger.set_level(config.INFO)
        
    #make environment
    env = get_environment(args.env_name)(verbose = args.verbose, manual = args.manual)
    env.seed(args.seed)
    set_global_seeds(args.seed)

    #load the agents
    ppo_models = load_all_models_with_names(env, start=args.start, stop=args.stop, step=args.step)
    total_agents = len(ppo_models)
    print(f"Loaded {total_agents} models in total.")

    #play all agents against each other
    for game_cell_i in range(total_agents):
        for game_cell_j in range(total_agents):
            agents = []

            agent_obj_1 = Agent(ppo_models[game_cell_i][1], ppo_models[game_cell_i][0])
            agent_obj_2 = Agent(ppo_models[game_cell_j][1], ppo_models[game_cell_j][0])
            # gonutsfordonuts perfectly blocks when playing against itself so 2 copies of the same agent is a bad idea
            # always use 1 copy of base that plays exploratively
            agent_obj_3 = Agent(ppo_models[0][1], ppo_models[0][0])
            
            agents.append(agent_obj_1)
            agents.append(agent_obj_2)
            agents.append(agent_obj_3)

            player_total_scores = []
            for p in range(3):
                player_total_scores.append(PlayerScore(agents[p].name))

            for k in range(args.games):
                total_rewards = {}
                
                total_rewards[agent_obj_1.id] = 0
                total_rewards[agent_obj_2.id] = 0
                total_rewards[agent_obj_3.id] = 0
                
                players = agents[:]

                # TODO: randomise player order without breaking records

                for p in players:
                    p.points = 0

                obs = env.reset()
                done = False

                game_str = f"game: ({game_cell_i},{game_cell_j}) players: {players[0].name},{players[1].name},{players[2].name}"
                logger.info(f"Playing {game_str}")

                for i, p in enumerate(players):
                    logger.info(f'Player {i+1} = {p.name}')

                while not done:

                    current_player = players[env.current_player_num]
                    env.render()
                    logger.info(f'\nCurrent player name: {current_player.name}')

                    logger.info(f'\n{current_player.name} model choices')
                                       
                    action = current_player.choose_action(env, choose_best_action = args.best, mask_invalid_actions = True)

                    obs, reward, done, _ = env.step(action)

                    for r, player in zip(reward, players):
                        total_rewards[player.id] += r
                        player.points += r
            
                env.render()

                write_results('results.csv', players, game_str, k, 0)

                for i, p in enumerate(player_total_scores):
                    p.scores.append(players[i].points)
            
            # write mean scoring for this set of tournament games
            for i, p in enumerate(player_total_scores):
                p.mean_score = mean(p.scores)

            write_tournament_results('tournament_results.csv', game_cell_i, game_cell_j, player_total_scores)

    env.close()

def cli() -> None:
    """Handles argument extraction from CLI and passing to main().
    Note that a separate function is used rather than in __name__ == '__main__'
    to allow unit testing of cli().
    """
    # Setup argparse to show defaults on help
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)

    parser.add_argument("--debug", "-d",  action = 'store_true', default = False
            , help="Show logs to debug level")
    parser.add_argument("--verbose", "-v",  action = 'store_true', default = False
            , help="Show observation on debug logging")
    parser.add_argument("--best", "-b", action = 'store_true', default = False
              , help="Play best moves (rather than sampling from policy prob distribution)")
    parser.add_argument("--manual", "-m",  action = 'store_true', default = False
            , help="Manual update of the game state on step")
    parser.add_argument("--seed", "-s",  type = int, default = 17
            , help="Random seed")
    parser.add_argument("--step", "-sx", type = int, default = 1
            , help="Step no through models to include in tournament")
    parser.add_argument("--start", "-st", type = int, default = 1
            , help="First model to include")
    parser.add_argument("--stop", "-sp", type = int, default = 1000
            , help="Last model - exclusive")
    parser.add_argument("--games", "-g", type = int, default = 1
                , help="Number of games to play)")
    parser.add_argument("--env_name", "-e",  type = str, default = 'TicTacToe'
            , help="Which game to play?")
    # Extract args
    args = parser.parse_args()

    # Enter main
    main(args)
    return


if __name__ == '__main__':
  cli()