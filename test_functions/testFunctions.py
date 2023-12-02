# -*- coding: utf-8 -*-
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from envs.simplePlant import SimplePlant
from agents import PerfectInfoAgent

def _test_agent(env: SimplePlant, agent, label, verbose=True):
    actions = []
    setup_costs = []
    lost_sales = []
    holding_costs = []
    demands = []
    observations = []
    # TEST
    if verbose: print("******")
    if verbose: print(f" {label} ")
    if verbose: print("******")
    obs = env.reset_time()
    if verbose: env.render()
    done = False
    while not done:
        action = agent.get_action(obs)
        if verbose: print(">>> action:", action)
        obs, _, done, info = env.step(action,verbose=verbose)
        demands.append(env.demand)
        actions.append(action)
        observations.append(obs)
        setup_costs.append(info['setup_costs'])
        lost_sales.append(info['lost_sales'])
        holding_costs.append(info['holding_costs'])
        if verbose: env.render()
    return actions, setup_costs, lost_sales, holding_costs , demands, observations

def plot_comparison(env, dict_results, col_dict={}):
    N_PLOT = 3
    if env.n_machines == 1:
        N_PLOT = 4

    plt.subplot(N_PLOT,1,1)
    for key in dict_results:
        plt.plot(dict_results[key]['setup_costs'], col_dict[key] if key in col_dict else "-b", label=key)
    plt.legend()
    plt.ylabel('setup')

    plt.subplot(N_PLOT,1,2)
    for key in dict_results:
        plt.plot(
            dict_results[key]['lost_sales'],
            col_dict[key] if key in col_dict else "-b",
            label=key
        )
    plt.legend()
    plt.ylabel('lost sales')
    
    plt.subplot(N_PLOT,1,3)
    for key in dict_results:
        plt.plot(
            dict_results[key]['holding_costs'],
            col_dict[key] if key in col_dict else "-b",
            label=key
        )
    plt.legend()
    plt.ylabel('holding costs')
    
    if env.n_machines == 1:
        plt.subplot(N_PLOT,1,4)
        for key in dict_results:
            plt.plot(
                dict_results[key]['actions'],
                col_dict[key] if key in col_dict else "-b",
                label=key
            )    
        plt.legend()
        plt.ylabel('action')
    plt.show()

def test_agents(env: SimplePlant, agents: list, n_reps: int =10, use_benchmark_PI=True, verbose=False, setting_sol_method = ''):
    dict_res = {}
    key_files = ['actions','setup_costs','lost_sales','holding_costs','demands','observations']
    for key, _ in agents:
        for key_file in key_files:
            dict_res[(key,key_file)] = []
        dict_res[(key,'costs')] = 0
    if use_benchmark_PI:
        for key_file in key_files:
            dict_res[('PI',key_file)] = []
        dict_res[('PI','costs')] = 0

    for _ in tqdm(range(n_reps)):
        # create a new scenarios
        env.reset()
        if use_benchmark_PI:
            pi_agent = PerfectInfoAgent(env, setting_sol_method)
            actions, setup_costs, lost_sales, holding_costs, demands, observations = _test_agent(
                env, pi_agent, label='PI', verbose=verbose
            )
            test_output = [actions, setup_costs, lost_sales, holding_costs, demands, observations]
            # appends the data from the tests
            
            for i in range(len(test_output)):
                dict_res[('PI',key_files[i])].append(test_output[i])
            
            if verbose: print("***")
            key = 'PI'
            dict_res[(key,'costs')] += (sum(setup_costs) / n_reps)
            dict_res[(key,'costs')]  += (sum(lost_sales) / n_reps)
            dict_res[(key,'costs')]  += (sum(holding_costs) / n_reps)

        # run each agent on the same scenario
        for key, agent in agents:
            actions, setup_costs, lost_sales, holding_costs,demands , observations = _test_agent(
                env, agent, label=key, verbose=verbose
            )
            test_output = [actions, setup_costs, lost_sales, holding_costs,demands , observations]
            dict_res[(key,'costs')]  += (sum(setup_costs) / n_reps)
            dict_res[(key,'costs')]  += (sum(lost_sales) / n_reps)
            dict_res[(key,'costs')]  += (sum(holding_costs) / n_reps)
            if verbose: print("***")
            for i in range(len(test_output)):
                dict_res[(key,key_files[i])].append(test_output[i])
    # saves the files from the tests
    save_files(
        dict_res,
        setting_sol_method
    )
    
    return dict_res

def save_files(dict_res,setting_sol_method):
    try:
        experiment_name = setting_sol_method['experiment_name']
        for key in dict_res:
            try:info = np.stack(dict_res[key])
            except:info = dict_res[key]
            np.save(
                os.path.join(
                    'results',
                    f'{key[0]}_{experiment_name}_{key[1]}_test'
                ),
                info
            )
    except:
        print('Fail to save the files.')
