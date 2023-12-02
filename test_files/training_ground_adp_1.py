# -*- coding: utf-8 -*-
import json
import random
import numpy as np
from envs import *
import logging
from agents import ValueIteration, AdpAgentHD1, StochasticProgrammingAgent
# from agents.adpAgentHD2 import AdpAgentHD2
from agents.adpAgentHD3 import AdpAgentHD3
from test_functions import *
from scenarioManager.stochasticDemandModel import StochasticDemandModel

np.random.seed(10)
random.seed(10)

log_name = "./logs/training_ground_adp_1.log"
logging.basicConfig(
    filename=log_name,
    format='%(message)s',
    level=logging.INFO,
    filemode='w'
)

# experiment_cfg = 'setting_10items_5machines'
# experiment_cfg = 'setting_6items_3machines'
# experiment_cfg = 'I_I50xM10xT7'
# experiment_cfg = 'setting_4items_2machines'
# experiment_cfg = 'setting_easy'
experiment_cfg = 'setting_2items_VI_binomial_3'
# experiment_cfg = 'setting_2items_VI'

# experiment_cfg = 'setting_100items_20machines'
# experiment_cfg = 'I_I50xM10xT10'
# experiment_cfg = 'setting_2items_VI'

fp = open(f"./cfg_env/{experiment_cfg}.json", 'r')
settings = json.load(fp)
fp.close()
settings['time_horizon'] = 10

stoch_model = StochasticDemandModel(settings)

env = SimplePlant(settings, stoch_model)
# env.plot_production_matrix()

fp = open(f"./cfg_sol/sol_setting.json", 'r')
setting_sol_method = json.load(fp)
fp.close()
# setting_sol_method["branching_factors"] = [4, 2]
agents = []

# vi_agent = ValueIteration(env, setting_sol_method)
# vi_agent.learn(iterations=1000)
# # vi_agent.plot_policy()
# # vi_agent.plot_value_function()
# agents.append(
#     ("VI", vi_agent)
# )
# setting_sol_method['regressor_name'] = 'plain_matrix_I2xM1'
setting_sol_method['regressor_name'] = 'matrix_independent'

# setting_sol_method['regressor_name'] = 'Random Forest'
adphd_agent1 = AdpAgentHD3(env, setting_sol_method)
# I_I50xM10xT7] MS: 
# setting_10items_5machines] ADP_H: 420.50 (SI) / 500 (con inv cost) / 445 expected inv MS: 471
# 6items_3machines] ADP_H: 50.80 MS: 60.0
# 4items_2machines] ADP_H: 29.15 MS: 37.48
# setting_2items_VI] ADP_H: 6.27 VI: 5.96 MS: 6.92
# easy] ADP_H: 7.6 MS: 8.13

"""
TODO: - wise learn
- con anche gli stati migliora ma c'è bisogno di un peso relativo all'impatto! 
"""
adphd_agent1.learn(epochs=100) # con 100 performa meglio che con 1000 su 4 items 2 machines.

# adphd_agent1.plot_policy()
# adphd_agent1.post_decision_value_function.V

# a = adphd_agent1._compute_best_action(
#     {
#         'inventory_level': [1, 1],
#         'machine_setup': [0]
#     }
# )
# print(a)
agents.append(
    ("ADP_H", adphd_agent1)
)

stoch_agent = StochasticProgrammingAgent(
    env,
    setting_sol_method
)
agents.append(
    ("RP", stoch_agent)
)

dict_res = test_agents(
    env,
    agents=agents,
    n_reps=30,
    use_benchmark_PI=False,
    verbose=False,
    setting_sol_method = setting_sol_method
)

for key,_ in agents:
    cost = dict_res[key,'costs']
    print(f'model {key}: {cost}')
    setup_costs = np.average([sum(ele) for ele in dict_res[key,'setup_costs']])
    lost_sales = np.average([sum(ele) for ele in dict_res[key,'lost_sales']])
    holding_costs = np.average([sum(ele) for ele in dict_res[key,'holding_costs']])
    print(f"\t s: {setup_costs:.2f} - l: {lost_sales:.2f} - h: {holding_costs:.2f}")
