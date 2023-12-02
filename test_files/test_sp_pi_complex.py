# -*- coding: utf-8 -*-
import os
import gc
import json
import random
import numpy as np
from envs import *
from agents import *
from test_functions import *
from scenarioManager.stochasticDemandModel import StochasticDemandModel


if __name__ == '__main__':

    np.random.seed(1)
    random.seed(10)
    
    experiment_name = '100items_10machines' # we set the experiment using the available files in cfg
    fp = open(f"./cfg_env/setting_{experiment_name}.json", 'r')
    settings = json.load(fp)
    fp.close()
    settings['time_horizon'] = 10
    
    stoch_model = StochasticDemandModel(settings)
    
    env = SimplePlant(settings, stoch_model)
    
    setting_sol_method = {
        'discount_rate': 0.9,
        'experiment_name': experiment_name,
        'parallelization': True,
        'model_name': 'A2C',
        'multiagent':False,
        'branching_factors': [4, 4, 4],
    }
    agents = []
    
    nreps = 10
    
    stoch_agent = StochasticProgrammingAgent(
        env,
        setting_sol_method
    )
    agents.append(("RP", stoch_agent))
    
    setting_sol_method['model_name'] = 'A2C'
    a2c_agent = StableBaselineAgent(
        env,
        setting_sol_method
    )
    #a2c_agent.learn(epochs=2000) # Each ep with 200 steps
    BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_A2C_{experiment_name}','best_model')
    a2c_agent.load_agent(BEST_MODEL_DIR)
    agents.append(("A2C", a2c_agent))
    
    dict_res = test_agents(
        env,
        agents=agents,
        n_reps=nreps,
        setting_sol_method = setting_sol_method
    )
    
    for key,_ in agents:
        cost = dict_res[key,'costs']
        print(f'\n Cost in {nreps} repetitions for the model {key}: {cost}')
    cost = dict_res['PI','costs']
    print(f'\n Cost in {nreps} repetitions for the model PI: {cost}')
