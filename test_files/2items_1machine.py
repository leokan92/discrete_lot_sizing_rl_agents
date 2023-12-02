# -*- coding: utf-8 -*-
# import os
# import gc
import json
import random
import numpy as np
from envs import *
from agents import ValueIteration#, StochasticProgrammingAgent, AdpAgentHD3
# from agents import *
# from agents.adpAgentHD3 import AdpAgentHD3 as AdpAgentHD
# from test_functions import *
from scenarioManager.stochasticDemandModel import StochasticDemandModel
import logging


if __name__ == '__main__':
    log_name = "./logs/2items_1machine.log"
    logging.basicConfig(
        filename=log_name,
        format='%(message)s',
        level=logging.INFO,
        filemode='w'
    )
    # Setting the seeds
    np.random.seed(1)
    random.seed(10)
    # Environment setup load:
    experiment_name = '2items_VI_binomial_1' # we set the experiment using the available files in cfg
    fp = open(f"./cfg_env/setting_{experiment_name}.json", 'r')
    settings = json.load(fp)
    fp.close()
    
    # Models setups:
    stoch_model = StochasticDemandModel(settings)
    settings['time_horizon'] = 20
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
    training_epochs_multiagent = 5000
    
    setting_sol_method['parallelization'] = True # Employed to accelerate the RL training (5 jobs in paralle - can be changged in the Stablebaseline agent directly)
    env = SimplePlant(settings, stoch_model)
    
    # Number of test execution (number of complet environment iterations)
    nreps = 100
    
    #########################################################################
    # # RL A2C
    # env = SimplePlant(settings, stoch_model)
    # setting_sol_method['model_name'] = 'A2C'
    # base_model_name = 'A2C'
    # rl_agent = StableBaselineAgent(
    #     env,
    #     setting_sol_method
    # )
    
    # #rl_agent.learn(epochs=training_epochs_RL) # Each ep with 200 steps
    
    # #load best agent before appending in the test list
    # BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
    # rl_agent.load_agent(BEST_MODEL_DIR)
    # agents.append(("A2C", rl_agent))
    
    #########################################################################
    # # RL PPO
    # setting_sol_method['model_name'] = 'PPO'
    # base_model_name = 'PPO'
    # rl_agent = StableBaselineAgent(
    #     env,
    #     setting_sol_method
    # )
    # #rl_agent.learn(epochs=training_epochs_RL) # Each ep with 200 steps
    
    # #load best agent before appending in the test list
    # BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
    # rl_agent.load_agent(BEST_MODEL_DIR) # For training purposes
    # agents.append(("PPO", rl_agent))
    
    
    ###########################################################################
    # #PSO
    
    # pso_agent = PSOagent(env)
    # #pso_agent.learn(iterations = 50, experiment_name = experiment_name)
    
    # pso_agent.load(experiment_name = experiment_name)
    # agents.append(
    #     ("PSO", pso_agent)
    # )
    ###########################################################################
    # VI
    settings['demand_distribution']['vals'] = [0,1,2,3,4, 5]
    settings['demand_distribution']['probs'] = [0.07, 0.25, 0.34, 0.23, 0.07, 0.01]
    vi_agent = ValueIteration(env, setting_sol_method)
    vi_agent.learn(iterations=1000)
    agents.append(
        ("VI", vi_agent)
    )
    vi_agent.plot_policy()
    ###########################################################################
    # RP
    # stoch_agent = StochasticProgrammingAgent(
    #     env,
    #     setting_sol_method
    # )
    # agents.append(("RP", stoch_agent))
    ###########################################################################
    # ADPHS
    # setting_sol_method['dict_obs'] = False
    # settings['dict_obs'] = False
    # setting_sol_method['parallelization'] = True
    # env = SimplePlant(settings, stoch_model)
    
    # adphd_agent = AdpAgentHD(env, setting_sol_method)
    # adphd_agent.learn(epochs=100) # does not have load method yet
    # agents.append(
    #     ("ADPHS", adphd_agent)
    # )
    # #########################
    setting_sol_method['regressor_name'] = 'matrix_independent'
    env = SimplePlant(settings, stoch_model)
    adphd_agent = AdpAgentHD3(env, setting_sol_method)
    adphd_agent.learn(epochs=500)
    adphd_agent.plot_policy()
    agents.append(
        ("ADPHS3", adphd_agent)
    )
    #########################################################################
    #TESTING
    env = SimplePlant(settings, stoch_model)
    dict_res = test_agents(
        env,
        agents=agents,
        n_reps=nreps,
        setting_sol_method = setting_sol_method,
        use_benchmark_PI=True
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
    # del env
    # gc.collect()

