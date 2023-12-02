# -*- coding: utf-8 -*-
import json
import random
import numpy as np
from agents import *
from envs.simplePlant import SimplePlant
from scenarioManager.stochasticDemandModel import StochasticDemandModel


# np.random.seed(1)
# random.seed(1)

experiment_name = '2items_VI'
fp = open(f"./cfg_env/setting_{experiment_name}.json", 'r')
settings = json.load(fp)
fp.close()
stoch_model = StochasticDemandModel(settings)

settings['time_horizon'] = 10

env = SimplePlant(settings, stoch_model)

fp = open(f"./cfg_sol/sol_setting.json", 'r')
setting_sol_method = json.load(fp)
fp.close()

adp_agent = AdpAgent(
    env,
    setting_sol_method,
)
adp_agent.learn(epochs=10)
adp_agent.plot_policy()

done = False
obs = env.reset_time()
reward_tot = 0
while not done:
    # print(obs)
    action = adp_agent.get_action(obs)
    # print(f"action: {action}")
    obs, reward, done, info = env.step(action)
    # print(f">> {info} -> {reward}")
    reward_tot += reward
print(reward_tot)

# VI
vi_agent = ValueIteration(env, setting_sol_method)
vi_agent.learn(iterations=100)
vi_agent.plot_policy()
done = False
obs = env.reset_time()
reward_tot = 0
while not done:
    # print(obs)
    action = vi_agent.get_action(obs)
    # print(f"action: {action}")
    obs, reward, done, info = env.step(action)
    # print(f">> {info} -> {reward}")
    reward_tot += reward
print(reward_tot)


ppo_agent = StableBaselineAgent(env, setting_sol_method)
ppo_agent.learn(epochs=1000)
ppo_agent.plot_policy()
done = False
obs = env.reset_time()
reward_tot = 0
while not done:
    # print(obs)
    action = ppo_agent.get_action(obs)
    # print(f"action: {action}")
    obs, reward, done, info = env.step(action)
    # print(f">> {info} -> {reward}")
    reward_tot += reward
print(reward_tot)
