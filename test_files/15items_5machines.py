# 
# -*- coding: utf-8 -*-
import os
import gc
import json
import random
import numpy as np
from envs import *
from agents import StochasticProgrammingAgent, AdpAgentHD3
from agents import StableBaselineAgent, MultiAgentRL, EnsembleAgent, PerfectInfoAgent,PSOagent,AdpAgentHD
from test_functions import *
from scenarioManager.stochasticDemandModel import StochasticDemandModel


if __name__ == '__main__':
    
    # Setting the seeds
    np.random.seed(1)
    random.seed(10)
    # Environment setup load:
    experiment_name = '15items_5machines' # we set the experiment using the available files in cfg
    fp = open(f"./cfg_env/setting_{experiment_name}.json", 'r')
    settings = json.load(fp)
    fp.close()
    
    # Models setups:
    stoch_model = StochasticDemandModel(settings)
    settings['time_horizon'] = 10
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
    training_epochs_multiagent = 20000
    
    
    setting_sol_method['parallelization'] = True
    env = SimplePlant(settings, stoch_model)
    
    # Number of test execution (number of complet environment iterations)
    nreps = 100
    
    #########################################################################
    # RL A2C
    env = SimplePlant(settings, stoch_model)
    setting_sol_method['model_name'] = 'A2C'
    base_model_name = 'A2C'
    rl_agent = StableBaselineAgent(
        env,
        setting_sol_method
    )
    
    #rl_agent.learn(epochs=training_epochs_RL) # Each ep with 200 steps
    
    #load best agent before appending in the test list
    BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
    rl_agent.load_agent(BEST_MODEL_DIR)
    agents.append(("A2C", rl_agent))
    
    #########################################################################
    # RL PPO


    setting_sol_method['model_name'] = 'PPO'
    base_model_name = 'PPO'
    ppo_agent = StableBaselineAgent(
        env,
        setting_sol_method
    )
    #rl_agent.learn(epochs=training_epochs_RL) # Each ep with 200 steps
    
    #load best agent before appending in the test list
    BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
    ppo_agent.load_agent(BEST_MODEL_DIR) # For training purposes
    agents.append(("PPO", ppo_agent))
    
    ###########################################################################
    #PSO
    
    pso_agent = PSOagent(env)
    pso_agent.learn(iterations = 50, experiment_name = experiment_name)
    
    pso_agent.load(experiment_name = experiment_name)
    agents.append(
        ("PSO", pso_agent)
    )
    ###########################################################################
    # RP
    stoch_agent = StochasticProgrammingAgent(
        env,
        setting_sol_method
    )
    agents.append(("RP", stoch_agent))
    ###########################################################################
    # ADP
    setting_sol_method['dict_obs'] = False
    settings['dict_obs'] = False
    setting_sol_method['parallelization'] = True
    env = SimplePlant(settings, stoch_model)
    
    adphd_agent = AdpAgentHD(env, setting_sol_method)
    adphd_agent.learn(epochs=100) # does not have load method yet
    agents.append(
        ("ADPHS", adphd_agent)
    )
    ###########################################################################
    # ADP NEW VERSION
    setting_sol_method['regressor_name'] = 'matrix_independent'
    env = SimplePlant(settings, stoch_model)
    adphd_agent_3 = AdpAgentHD3(env, setting_sol_method)
    adphd_agent_3.learn(epochs=100) # does not have load method yet
    agents.append(
        ("ADPHS3", adphd_agent_3)
    )
    #########################################################################
    # Multi-agent RL - PPO single agent and PPO base agent
    
    setting_sol_method['parallelization'] = True
    
    single_agent_model = 'PPO'
    base_model_name = 'ADPHS'
    
    setting_sol_method['multiagent'] = False
    setting_sol_method['model_name'] = base_model_name # indivual RL model employed in the Multi-agent system
    setting_sol_method['experiment_name'] = experiment_name
    setting_sol_method['dict_obs'] = False
    env = SimplePlant(settings, stoch_model)
    
    
    
    # base_rl_agent = StableBaselineAgent(
    #     env,
    #     settings = setting_sol_method
    # )
    
    
    # BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
    # base_rl_agent.load_agent(BEST_MODEL_DIR) # For training purposes
    adphd_agent.model_name = base_model_name
    base_rl_agent = adphd_agent
    
    setting_sol_method['multiagent'] = True
    setting_sol_method['model_name'] = single_agent_model 
    setting_sol_method['dict_obs'] = True
    settings['dict_obs'] = setting_sol_method['dict_obs']
    env = SimplePlant(settings, stoch_model)
    multiagent = MultiAgentRL(settings,
                              stoch_model,
                              setting_sol_method, # indivual RL model employed in the Multi-agent system
                              base_rl_agent = base_rl_agent)
    

    multiagent.learn(epochs = training_epochs_multiagent)
    multiagent.load_agent()
    agents.append((f'MultiAgent_{base_model_name}_{single_agent_model}', multiagent))
    
    
    # #########################################################################
    # Ensemble agent - PPO SC
    
    # setting_sol_method['parallelization'] = True    
    # setting_sol_method['model_name'] = 'PPO'
    # setting_sol_method['dict_obs'] = False
    # settings['dict_obs'] = setting_sol_method['dict_obs']
    # env = SimplePlant(settings, stoch_model)
    # first_agent_name = 'PPO'
    # # base_model_name = 'PPO'
    # # first_agent = StableBaselineAgent(
    # #     env,
    # #     setting_sol_method
    # # )
    # # BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
    # # first_agent.load_agent(BEST_MODEL_DIR) # For training purposes
    # second_agent = adphd_agent
    # second_agent.model_name = "ADPHS"
    # setting_sol_method['multiagent'] = True
    # setting_sol_method['dict_obs'] = True
    # settings['dict_obs'] = setting_sol_method['dict_obs']
    # env = SimplePlant(settings, stoch_model)
    # ensemble_agent = EnsembleAgent(settings,
    #                                stoch_model,
    #                                setting_sol_method, # indivual RL model employed in the Multi-agent system
    #                                first_agent = second_agent,
    #                                second_agent = second_agent)
    # ensemble_agent.learn(int(training_epochs_multiagent))
        
    # ensemble_agent.load_agent()
    # agents.append(("Ensemble", ensemble_agent))
    
    #########################################################################
    #TESTING
    settings['dict_obs'] = False
    setting_sol_method['multiagent'] = False
    setting_sol_method['dict_obs'] = False
    env = SimplePlant(settings, stoch_model)
    setting_sol_method['experiment_name'] = experiment_name
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
    del env
    gc.collect()
