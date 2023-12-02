# -*- coding: utf-8 -*-
import os
import copy
import time
import math
from envs import *
import numpy as np
import pandas as pd
from tqdm import tqdm
from alns.stop import *
from alns.accept import *
from alns.weights import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from agents.utils.regressor_lib import RegressorLib


class AdpAgentEasy():
    def __init__(self, env: SimplePlant, settings: dict):
        super(AdpAgentEasy, self).__init__()
        self.env = copy.deepcopy(env)
        self.env_for_exp = copy.deepcopy(env)
        self.discount = settings['discount_rate']
        self.post_decision_value_function = RegressorLib(settings['regressor_name'])
        # create array of valid actions
        self.valid_actions = []
        for m in range(self.env.n_machines):
            self.valid_actions.append([0])
            self.valid_actions[-1].extend(
                list(np.nonzero(self.env.machine_production_matrix[m])[0] + 1)
            )
        # casting to np array
        self.machine_production = np.array(self.env.machine_production_matrix)
        self.setup_costs = np.array(self.env.setup_costs)
        self.setup_loss = np.array(self.env.setup_loss)
        # starting to solve
        self.gamma = 1
        self.alpha = 1
        self.n_episode = 0

    def get_action(self, obs):
        _, action, _ = self._compute_best_action(obs)
        return action

    def train_policy(self):
        # SETTING GAMMA
        avg_demand = np.mean(self.env.scenario_demand, axis=1)
        max_prod = np.max(self.machine_production, axis=0)
        max_T = int(np.ceil(max(max_prod / avg_demand)))

        best_perf_gamma = np.inf
        best_gamma = 0
        for gamma in range(1, 2 * max_T):
            self.gamma = gamma
            perf = self.simulate_episodes(n_episodes = 10)
            if perf < best_perf_gamma:
                print(f"Better gamma found: {gamma}")
                best_perf_gamma = perf
                best_gamma = gamma
        self.gamma = best_gamma
        # SETTING ALPHA
        best_perf_alpha = np.inf
        best_alpha = 0
        for alpha in [0.5, 0.1, 0.01, 0]:
            self.alpha = alpha
            perf = self.simulate_episodes(n_episodes = 10)
            if perf < best_perf_alpha:
                print(f"Better alpha found: {alpha}")
                best_perf_alpha = perf
                best_alpha = alpha
        self.alpha = best_alpha

    def simulate_episodes(self, n_episodes, initial_obs=None, initial_action=None):
        tot_cost = 0
        for _ in range(n_episodes):
            if initial_obs:
                # modify initial_inventory since it is set in reset time
                for i in range(self.env_for_exp.n_items):
                    self.env_for_exp.initial_inventory[i] = initial_obs['inventory_level'][i]
                for m in range(self.env_for_exp.n_machines):
                    self.env_for_exp.machine_initial_setup[m] = initial_obs['machine_setup'][m]
                obs = self.env_for_exp.reset_time()
            else:
                obs = self.env_for_exp.reset()
            partial_cost = 0
            for t in range(self.env.T):
                if initial_action and t == 0: # WHY?
                    action = initial_action
                else:
                    action = self.get_action(obs)
                obs, cost, _, _ = self.env_for_exp.step(action)
                partial_cost += cost * (self.discount)**t
            tot_cost += partial_cost / n_episodes
        return tot_cost

    def _compute_best_action_rule(self, obs):
        # print(f"inventory_level: {obs['inventory_level']}")
        # print(f"machine_setup: {obs['machine_setup']}")
        avg_demand = np.mean(self.env.scenario_demand, axis=1)
        self.runout = obs['inventory_level'] / (avg_demand + 0.001)
        best_action = [0] * self.env.n_machines
        self.eligible_items = []
        # metto due parametri gamma e alpha

        for eligible_item in np.where(self.runout < self.gamma)[0]:
            # TODO: cambiare la formula magari con frazioni, come su MATLAB
            priority = 1 / (self.runout[eligible_item] + 0.01)
            priority += self.alpha * (self.env.lost_sales_costs[eligible_item] - self.env.holding_costs[eligible_item])
            self.eligible_items.append((eligible_item, priority))
        sorted_eligible_items = sorted(self.eligible_items, key=lambda tup: tup[1])
        for idx_item, _ in sorted_eligible_items:
            # if one machine is producting idx_item then continue
            if idx_item + 1 in obs['machine_setup']:
                for i, ele in enumerate(obs['machine_setup']):
                    # idx_item + 1 and not idx_item since in the action we have also the do nothing option.
                    if ele == idx_item + 1:
                        best_action[i] = idx_item + 1
            else:
                # if nobody is producing I activate the machine less expensive
                machine_production_lst = self.machine_production[:, idx_item]
                setup_costs_lst = self.setup_costs[:, idx_item]
                setup_loss_lst = self.setup_loss[:, idx_item]
                kpi = setup_costs_lst + self.env.holding_costs[idx_item] * (machine_production_lst - setup_loss_lst - avg_demand[idx_item])
                # another good policy can be:
                # kpi = setup_costs_lst / (machine_production_lst + 0.001)
                # remove machine with 0 production
                kpi[machine_production_lst == 0] = np.inf
                best_machine = np.argmin(kpi)
                best_action[best_machine] = idx_item + 1
        return best_action

    def _evaluate_action(self, action, obs):
        temp_dem = self.env.stoch_model.generate_scenario(n_time_steps=100)
        tot_lost_sales = 0
        tot_holding_costs = 0
        tot_future = 0
        setup_costs = np.zeros(self.env.n_machines)
        machine_setup = [ele for ele in obs['machine_setup']]
        original_inventory_level = [ele for ele in obs['inventory_level']]
        self.env._post_decision(
            action,
            machine_setup,
            original_inventory_level,
            setup_costs
        )

        for ii in range(10):
            lost_sales = np.zeros(self.env.n_items)
            holding_costs = np.zeros(self.env.n_items)
            inventory_level = [ele for ele in original_inventory_level]
            self.env._satisfy_demand(
                lost_sales,
                inventory_level,
                temp_dem[:,ii],
                # d,
                holding_costs
            )
            tot_lost_sales += sum(lost_sales)
            tot_holding_costs += sum(holding_costs)
            new_obs = {
                'inventory_level': inventory_level,
                'machine_setup': machine_setup
            }
            tmp, tmp_act, _ = self._compute_best_action(
                new_obs,
                epsilon_prob=0
            )
            tot_future += tmp
        tot_lost_sales /= 100
        tot_holding_costs /= 100
        tot_future /= 100
        obs, cost, _, info = self.env.step(action, verbose=False)
        tot_cost = info['setup_costs'] + tot_lost_sales + tot_holding_costs

        return tot_cost
    
    def _compute_best_action(self, obs):
        # small neiborhood search
        best_action = self._compute_best_action_rule(obs)

        best_val, best_post_data = self._evaluate_action(best_action, obs)
        action = [ele for ele in best_action] # [0] * self.env.n_machines
        # TODO: time consuming
        for m in range(self.env.n_machines):
            for i in self.valid_actions[m]:
                action[m] = i
                tmp, pd_data = self._evaluate_action(action, obs)    
                if tmp < best_val:
                    best_val = tmp
                    best_action = [ele for ele in action]
                    best_post_data = pd_data

    def learn(self, epochs = 10):
        self.train_policy()

