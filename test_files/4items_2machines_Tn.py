# -*- coding: utf-8 -*-
import os
import gc
import json
import time
import random
import numpy as np
from envs import *
from agents import *
from test_functions import *
from scenarioManager.stochasticDemandModel import StochasticDemandModel


if __name__ == '__main__':
    for t in [5,10,20,40,80,160]:
        # Setting the seeds
        np.random.seed(1)
        random.seed(10)
        # Environment setup load:
        experiment_name = '4items_2machines' # we set the experiment using the available files in cfg
        fp = open(f"./cfg_env/setting_{experiment_name}.json", 'r')
        settings = json.load(fp)
        fp.close()
        
        # Models setups:
        stoch_model = StochasticDemandModel(settings)
        settings['time_horizon'] = t
        env = SimplePlant(settings, stoch_model)
        settings['dict_obs'] = False
        setting_sol_method = {
            'discount_rate': 0.9,
            'experiment_name': experiment_name,
            'parallelization': False,
            'model_name': 'A2C',
            'multiagent':False,
            'branching_factors': [4, 2, 2],
            'dict_obs': False # To be employed if dictionary observations are necessary
        }
        # Parameters for the ADPHS:
        setting_sol_method['regressor_name'] = 'plain_matrix_I2xM1'
        setting_sol_method['discount_rate'] = 0.9
        agents = []
        # Parameters for the RL:
       
        training_epochs_RL = 50000
        training_epochs_multiagent = 7000
        
        
        setting_sol_method['parallelization'] = True
        env = SimplePlant(settings, stoch_model)
        
        # Number of test execution (number of complet environment iterations)
        nreps = 10
        start = time.time()
        ###########################################################################
        # RP
        stoch_agent = StochasticProgrammingAgent(
            env,
            setting_sol_method
        )
        agents.append(("RP", stoch_agent))


        #########################################################################
        #TESTING
        env = SimplePlant(settings, stoch_model)
        dict_res = test_agents(
            env,
            agents=agents,
            n_reps=nreps,
            setting_sol_method = setting_sol_method,
            use_benchmark_PI=False
        )
        
        for key,_ in agents:
            cost = dict_res[key,'costs']
            print(f'\n Cost in {nreps} iterations for the model {key}: {cost}')
        try:
            cost = dict_res['PI','costs']
            print(f'\n Cost in {nreps} repetitions for the model PI: {cost}')
        except:
            pass
                
        #del multiagent
        del env
        gc.collect()
        print(f'\nExecution Time. Elepsed time (T = {t}): {round(time.time()-start,1)}s')
