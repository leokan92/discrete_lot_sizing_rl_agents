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

# experiment_cfg = 'setting_10items_5machines'
# experiment_cfg = 'setting_6items_3machines'
experiment_cfg = 'I_I50xM10xT7'
# experiment_cfg = 'setting_4items_2machines'
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
# vi_agent.plot_policy()
# # vi_agent.plot_value_function()
# agents.append(
#     ("VI", vi_agent)
# )
# setting_sol_method['regressor_name'] = 'plain_matrix_I2xM1'
# setting_sol_method['discount_rate'] = 0.9

# setting_sol_method['regressor_name'] = 'Random Forest'
adphd_agent = AdpAgentHD(env, setting_sol_method)
agents.append(
    ("ADPHS", adphd_agent)
)
# adphd_agent.train_policy()
# adphd_agent._compute_best_action_rule(
#     {'inventory_level': [0.0, 0.0], 'machine_setup': [1]}
# )
adphd_agent.learn(epochs=1000)
# adphd_agent.plot_post_decision_value_function()
# adphd_agent.plot_policy()

if True:
    # # PPO
    # ppo_agent = StableBaselineAgent(
    #     env,
    #     setting_sol_method
    # )
    # ppo_agent.learn(epochs=100) # Each ep with 200 steps
    # agents.append(
    #     ("PPO", ppo_agent)
    # )
    # SP
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
        n_reps=1,
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
    # cost = dict_res['PI','costs']
    # print(f'model PI: {cost}')
