

import os
import sys
import random
import csv
import time
import numpy as np
import math

from mpi4py import MPI

from shutil import rmtree
from stable_baselines.ppo1 import PPO1
from stable_baselines.common.policies import MlpPolicy

from collections import OrderedDict

from utils.register import get_network_arch

import config

from stable_baselines import logger

def write_tournament_results(filename, x, y, scores):

    fieldnames = OrderedDict([('x', None), ('y', None), ('model0', None), ('score0', None), ('model1', None), ('score1', None), ('model2', None), ('score2', None)])
    out = { 'x': x, 'y': y}
    
    for i, p in enumerate(scores):
        out[f'model{i}'] = p.name
        out[f'score{i}'] = p.mean_score

    resultsfile = f"{config.RESULTSPATH}/{filename}"

    logger.info(f"Writing tournament result file: {resultsfile}")

    if not os.path.exists(resultsfile):
        with open(resultsfile,'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

    with open(resultsfile,'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow(out)

def write_results(filename, players, game, games, episode_length):
    
    out = {'game': game
    , 'games': games
    , 'episode_length': episode_length }

    for i, p in enumerate(players):
        out[f'p{i}'] = p.name
        out[f'p{i}_points'] = p.points

    resultsfile = f"{config.RESULTSPATH}/{filename}"

    if not os.path.exists(config.RESULTSPATH):
        with open(resultsfile,'a') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=out.keys())
            writer.writeheader()

    with open(resultsfile,'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=out.keys())
        writer.writerow(out)


def load_model(env, name):

    filename = os.path.join(config.MODELDIR, env.name, name)
    if os.path.exists(filename):
        logger.info(f'Loading {name}')
        cont = True
        while cont:
            try:
                ppo_model = PPO1.load(filename, env=env)
                cont = False
            except Exception as e:
                time.sleep(5)
                print(e)
    
    elif name == 'base.zip':
        cont = True
        while cont:
            try:
                
                rank = MPI.COMM_WORLD.Get_rank()
                if rank == 0:
                    ppo_model = PPO1(get_network_arch(env.name), env=env)
                    logger.info(f'Saving base.zip PPO model...')
                    ppo_model.save(os.path.join(config.MODELDIR, env.name, 'base.zip'))
                else:

                    ppo_model = PPO1.load(os.path.join(config.MODELDIR, env.name, 'base.zip'), env=env)

                cont = False
            except IOError as e:
                sys.exit(f'Check zoo/{env.name}/ exists and read/write permission granted to user')
            except Exception as e:
                logger.error(e)
                time.sleep(2)
                
    else:
        raise Exception(f'\n{filename} not found')
    
    return ppo_model


def load_all_models(env):
    modellist = [f for f in os.listdir(os.path.join(config.MODELDIR, env.name)) if f.startswith("_model")]
    modellist.sort()

    models = [load_model(env, 'base.zip')]
    for model_name in modellist:
        models.append(load_model(env, name = model_name))
    return models

def load_all_models_with_names(env, start=None, stop=None, step=None):
    modellist = [f for f in os.listdir(os.path.join(config.MODELDIR, env.name)) if f.startswith("_model")]
    modellist.sort()
    print(f"Length of model list {len(modellist)}")

    modellist = modellist[slice(start, stop, step)]
    
    models = [(load_model(env, 'base.zip'), 'base')]
    for model_name in modellist:
        models.append((load_model(env, name = model_name), model_name))
    return models

def get_best_model_name(env_name):
    modellist = [f for f in os.listdir(os.path.join(config.MODELDIR, env_name)) if f.startswith("_model")]
    
    if len(modellist)==0:
        filename = None
    else:
        modellist.sort()
        filename = modellist[-1]
        
    return filename

def get_model_stats(filename):
    if filename is None:
        generation = 0
        timesteps = 0
        best_rules_based = -np.inf
        best_reward = -np.inf
    else:
        stats = filename.split('_')
        generation = int(stats[2])
        best_rules_based = float(stats[3])
        best_reward = float(stats[4])
        timesteps = int(stats[5])
    return generation, timesteps, best_rules_based, best_reward


def reset_logs(model_dir):
    try:
        filelist = [ f for f in os.listdir(config.LOGDIR) if f not in ['.gitignore']]
        for f in filelist:
            if os.path.isfile(f):  
                os.remove(os.path.join(config.LOGDIR, f))

        for i in range(100):
            if os.path.exists(os.path.join(config.LOGDIR, f'tb_{i}')):
                rmtree(os.path.join(config.LOGDIR, f'tb_{i}'))
        
        open(os.path.join(config.LOGDIR, 'log.txt'), 'a').close()
    
        
    except Exception as e :
        print(e)
        print('Reset logs failed')

def reset_models(model_dir):
    try:
        filelist = [ f for f in os.listdir(model_dir) if f not in ['.gitignore']]
        for f in filelist:
            os.remove(os.path.join(model_dir , f))
    except Exception as e :
        print(e)
        print('Reset models failed')