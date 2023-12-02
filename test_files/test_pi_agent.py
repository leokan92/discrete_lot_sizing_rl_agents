# -*- coding: utf-8 -*-
import json
import random
import numpy as np
from envs import *
from agents import *
from test_functions import *
from scenarioManager.stochasticDemandModel import StochasticDemandModel

np.random.seed(10)
random.seed(10)

# experiment_name = '2items_VI'
experiment_name = '4items_2machines'
experiment_name = '100items_20machines'
experiment_name = '2items_VI_binomial_3'

fp = open(f"./cfg_env/setting_{experiment_name}.json", 'r')
settings = json.load(fp)
fp.close()
settings["time_horizon"] = 5

stoch_model = StochasticDemandModel(settings)

env = SimplePlant(settings, stoch_model)

fp = open(f"./cfg_sol/sol_setting.json", 'r')
setting_sol_method = json.load(fp)
fp.close()


# SP
stoch_agent = StochasticProgrammingAgent(
    env,
    setting_sol_method
)

obs = env.reset() # changes initial inventory and initial set up.
# pi_agent = PerfectInfoAgent(env, setting_sol_method)

done = False
reward_tot = 0
while not done:
    # print(obs)
    action = stoch_agent.get_action(obs)
    print(f"action: {action}")
    obs, reward, done, info = env.step(action, verbose=True)
    # print(f">> {info} -> {reward}")
    reward_tot += reward
print("STOCH", reward_tot)
print("******")
print("******")
print("******")

# done = False
# obs = env.reset_time()
# reward_tot = 0
# while not done:
#     # print(f"obs: {obs}")
#     action = pi_agent.get_action(obs)
#     # print(f"action: {action}")
#     obs, reward, done, info = env.step(action, verbose=True)
#     # print(f"\t {info} -> {reward}")
#     reward_tot += reward
# print("PI", reward_tot)
