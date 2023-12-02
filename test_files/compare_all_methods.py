# -*- coding: utf-8 -*-
import json
import random
import numpy as np
from envs import *
from agents import ValueIteration
# from test_functions import *
from scenarioManager.stochasticDemandModel import StochasticDemandModel

# np.random.seed(4)
# random.seed(4)

# experiment_name = '4items_2machines'
experiment_name = '2items_VI_binomial_2'

fp = open(f"./cfg_env/setting_{experiment_name}.json", 'r')
settings = json.load(fp)
fp.close()
settings["time_horizon"] = 10

stoch_model = StochasticDemandModel(settings)

env = SimplePlant(settings, stoch_model)

fp = open(f"./cfg_sol/sol_setting.json", 'r')
setting_sol_method = json.load(fp)
fp.close()
agents = []

# VI
# if experiment_name == '2items_VI':
vi_agent = ValueIteration(env, setting_sol_method)
vi_agent.learn(iterations=1000)
agents.append(
    ("VI", vi_agent)
)
vi_agent.plot_policy()

# # PPO
# ppo_agent = StableBaselineAgent(
#     env,
#     setting_sol_method
# )
# ppo_agent.learn(epochs=200) # Each ep with 200 steps
# agents.append(
#     ("PPO", ppo_agent)
# )


# # A2C
# setting_sol_method['model_name'] = 'A2C'
# a2c_agent = StableBaselineAgent(
#     env,
#     setting_sol_method
# )
# a2c_agent.learn(epochs=200) # Each ep with 200 steps
# agents.append(
#     ("A2C", a2c_agent)
# )

# # SP
# stoch_agent = StochasticProgrammingAgent(
#     env,
#     setting_sol_method
# )
# agents.append(
#     ("RP", stoch_agent)
# )

# # ADP
# if experiment_name == '2items_VI':
#     adp_agent = AdpAgent(
#         env,
#         setting_sol_method
#     )
#     adp_agent.learn(epochs=20)
#     agents.append(
#         ("ADP", adp_agent)
#     )

# nreps = 30

# dict_res = test_agents(
#     env,
#     agents=agents,
#     n_reps=nreps,
#     setting_sol_method = setting_sol_method
# )


# for key,_ in agents:
#     cost = dict_res[key,'costs']
#     print(f'Cost in {nreps} repetitions for the model {key}: {cost}')
# cost = dict_res['PI','costs']
# print(f'Cost in {nreps} repetitions for the model PI: {cost}')

