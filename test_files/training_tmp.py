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


cfgs = ['setting_4items_2machines', 'setting_6items_3machines', 'setting_10items_5machines']#, 'setting_100items_10machines']
for experiment_cfg in cfgs:
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
    agents = []

    setting_sol_method['regressor_name'] = 'plain_matrix_I2xM1'
    setting_sol_method['discount_rate'] = 0.9

    adphd_agent = AdpAgentHD(env, setting_sol_method)
    agents.append(
        ("ADPHS", adphd_agent)
    )
    adphd_agent.train_policy()

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
        n_reps=10,
        use_benchmark_PI=False,
        setting_sol_method = setting_sol_method
    )

    for key,_ in agents:
        cost = dict_res[key,'costs']
        print(f'{experiment_cfg}] model {key}: {cost} []')
        setup_costs = np.average([sum(ele) for ele in dict_res[key,'setup_costs']])
        lost_sales = np.average([sum(ele) for ele in dict_res[key,'lost_sales']])
        holding_costs = np.average([sum(ele) for ele in dict_res[key,'holding_costs']])
        print(f"\t s: {setup_costs:.2f} - l: {lost_sales:.2f} - h: {holding_costs:.2f}")

    '''
    setting_4items_2machines] model ADPHS: 58.209999999999994 []
            s: 7.10 - l: 46.00 - h: 5.11
                            ] model RP: 54.780000000000015 []
            s: 13.00 - l: 36.20 - h: 5.58
    setting_6items_3machines] model ADPHS: 94.53 []
            s: 8.70 - l: 78.80 - h: 7.03
                            ] model RP: 60.67 []
            s: 16.80 - l: 33.40 - h: 10.47
    setting_10items_5machines] model ADPHS: 102.38000000000002 []
         s: 24.00 - l: 60.50 - h: 17.88
                            ] model RP: 79.58999999999999 []
            s: 32.70 - l: 28.80 - h: 18.09
    '''
'''
import numpy as np
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

X = np.array(
    [
        [1],
        [2],
        [3],
        [4],
        [5],
        [6],
        [7],
        [8]
    ]
)
y = np.array([1,2,3,4,5,6,7,8])

regressor = RandomForestRegressor(n_estimators=50)
regressor.fit(X, y)  # warm_start=False
print(f"R2: {regressor.score(X, y)}")
output = regressor.predict(
    np.array(
        [1]
    ).reshape(1, -1)
).item()
print(output)
'''
