# -*- coding: utf-8 -*-
import json
import random
import numpy as np
from envs import *
from agents import *
from test_functions import *
from scenarioManager.stochasticDemandModel import StochasticDemandModel

np.random.seed(1)
random.seed(1)

experiment_name = '4items_2machines'
fp = open(f"./cfg_env/setting_{experiment_name}.json", 'r')
settings = json.load(fp)
fp.close()
settings["time_horizon"] = 5

stoch_model = StochasticDemandModel(settings)

env = SimplePlant(settings, stoch_model)

fp = open(f"./cfg_sol/sol_setting.json", 'r')
setting_sol_method = json.load(fp)
fp.close()
agents = []

# PPO
ppo_agent = StableBaselineAgent(
    env,
    setting_sol_method
)
ppo_agent.learn(epochs=100) # Each ep with 200 steps
agents.append(("PPO", ppo_agent))


# # A2C
# setting_sol_method['model_name'] = 'A2C'
# a2c_agent = StableBaselineAgent(
#     env,
#     setting_sol_method
# )
# a2c_agent.learn(epochs=10) # Each ep with 200 steps
# agents.append(("A2C", a2c_agent))

# SP
stoch_agent = StochasticProgrammingAgent(
    env,
    setting_sol_method
)
agents.append(("RP", stoch_agent))


nreps = 1
dict_res = test_agents(
    env,
    agents=agents,
    n_reps=nreps,
    verbose=True,
    setting_sol_method=setting_sol_method
)
for key,_ in agents:
    cost = dict_res[key,'costs']
    print(f'Cost in {nreps} repetitions for the model {key}: {cost}')
cost = dict_res['PI','costs']
print(f'Cost in {nreps} repetitions for the model PI: {cost}')

