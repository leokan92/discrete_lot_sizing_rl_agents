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
# from test_functions import _test_agent

class AdpAgentHD():
    def __init__(self, env: SimplePlant, settings: dict):
        super(AdpAgentHD, self).__init__()
        self.env = copy.deepcopy(env)
        self.env_for_exp = copy.deepcopy(env)
        self.discount = settings['discount_rate']
        self.post_decision_value_function = RegressorLib(
            settings['regressor_name'], env
        )
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
        self.alpha = np.zeros(6)
        self.alpha[0] = 1
        self.n_episodes_to_compare = 500
        self.envs = [copy.deepcopy(env) for i in range(self.n_episodes_to_compare)]
        for env in self.envs:
            env.reset()
        self.train_done = False

    def get_action(self, obs):
        # _, action, _ = self._compute_best_action(obs)
        action = self._compute_best_action_rule(obs)
        # TODO: post optimization?
        return action

    def train_policy(self):
        scenario_demand = self.env.stoch_model.generate_scenario(
            n_time_steps=1000
        )
        n_scenarios = scenario_demand.shape[1]
        max_demand = np.max(scenario_demand, axis=1)
        self.best_inv = np.zeros(self.env.n_items)
        empirical_distr = []
        for i in range(self.env.n_items):
            empirical_distr.append(
                np.zeros(int(max_demand[i]) + 1)
            )
            for ele in scenario_demand[i,:]:
                empirical_distr[i][int(ele)] += 1 / n_scenarios
            best_val = np.inf
            best_I = 0
            for I in range(self.env.max_inventory_level[i]):
                val = 0
                for d, p in enumerate(empirical_distr[i]):
                    # print(I, p, self.env.lost_sales_costs[i] * max(d-I, 0) + self.env.holding_costs[i] * max(I-d, 0))
                    val += p * (self.env.lost_sales_costs[i] * max(d-I, 0) + self.env.holding_costs[i] * max(I-d, 0))
                # print(I, val)
                if val < best_val:
                    best_val = val
                    best_I = I
            self.best_inv[i] = best_I

        def test_alpha(alpha):
            self.alpha = alpha
            return self.simulate_episodes(use_rule=True)

        alphas = [
            [4, 1, 0.5, 0.5, 1, 0.7],
            [4, 1, 0.5, 0.5, 1, 1.2],
            [3, 1, 0.5, 0.5, 1, 0.7],
            [3, 1.1, 1, 1, 1, 1],
            [3, 1.1, 0.8, 1, 1.2, 1],
            [3, 1.1, 1, 1.2, 1, 1],
            [3, 1.2, 1, 1, 1, 0.7],
            [3, 1.1, 1, 1, 1, 1.2],
            [2, 1.5, 0.5, 0.5, 1, 0.7],
            [1, 1.1, 0.5, 0.5, 1, 0.7],
            [1, 1.5, 0.5, 0.5, 1, 0.7],
            [1, 1, 1, 1, 1, 1],
            [5, 1, 0.5, 0.5, 1, 0.7],
            [5, 2, 0.5, 0.5, 1, 1.2],
            [6, 1, 0.5, 0.5, 1, 0.7],
            [6, 2, 0.5, 0.5, 1, 1.2],
            [7, 1, 0.5, 0.5, 1, 0.7],
            [7, 2, 0.5, 0.5, 1, 1.2],
            [8, 1, 0.5, 0.5, 1, 0.7],
            [8, 2, 0.5, 0.5, 1, 1.2],
            [9, 1, 0.5, 0.5, 1, 0.7],
            [9, 2, 0.5, 0.5, 1, 1.2],
            [10, 1, 0.5, 0.5, 1, 0.7],
            [10, 2, 0.5, 0.5, 1, 1.2],
        ]
        # runout time
        # safety per best_inv
        # improve if setup present
        # inventory_level < best_inv
        # if avg_demand > production
        # if setup_cost > estimated_holding_costs
        best_val = np.inf
        for alpha in alphas:
            ans = test_alpha(alpha)
            if ans < best_val:
                best_val = ans
                best_alpha = [ele for ele in alpha]
        print(best_val, best_alpha)
        self.alpha = best_alpha
        self.train_done = True

    def simulate_episodes(self, use_rule=False):
        tot_cost = 0
        for env_for_exp in self.envs:
            obs = env_for_exp.reset_time()
            partial_cost = 0
            done = False
            while not done:
                if use_rule:
                    action = self._compute_best_action_rule(obs)
                else:
                    action = self.get_action(obs)
                obs, cost, done, _ = env_for_exp.step(action)
                partial_cost += cost
            tot_cost += partial_cost / self.n_episodes_to_compare
        return tot_cost

    def _compute_best_action_rule(self, obs):
        # print(f"inventory_level: {obs['inventory_level']}")
        # print(f"machine_setup: {obs['machine_setup']}")
        avg_demand = np.mean(self.env.scenario_demand, axis=1)

        self.runout = obs['inventory_level'] / (avg_demand + 0.001)
        best_action = [0] * self.env.n_machines
        self.eligible_items = []
        # metto due parametri gamma e alpha
        list_items = np.union1d(
            np.where(self.runout < self.alpha[0])[0],
            np.where(obs['inventory_level'] < self.alpha[1] * self.best_inv[0])
        )

        for eligible_item in list_items: #np.where(obs['inventory_level'] < self.best_inv)[0]: #
            priority = self.env.lost_sales_costs[eligible_item] / (self.runout[eligible_item] + 0.01)
            priority += self.alpha[2] * (obs['machine_setup'].count(eligible_item + 1) >= 1)
            # if a machine is producing the item, higher priority 
            # if the inventory is lower than the best_I
            priority += self.alpha[3] * (obs['inventory_level'][eligible_item] <= self.best_inv[eligible_item])
            # if production is low respect the demand:
            priority += self.alpha[4] * avg_demand[eligible_item] / max(self.machine_production[:, eligible_item])
            self.eligible_items.append((eligible_item, priority))
        
        sorted_eligible_items = sorted(self.eligible_items, key=lambda tup: tup[1], reverse=True)

        # START PRODUCTION FOR ELIGIBLE ITEM
        for idx_item, _ in sorted_eligible_items:
            # if one machine is producting idx_item then continue
            if idx_item + 1 in obs['machine_setup']:
                for m, ele in enumerate(obs['machine_setup']):
                    # idx_item + 1 and not idx_item since in the action we have also the do nothing option.
                    if ele == idx_item + 1 and best_action[m] == 0:
                        best_action[m] = idx_item + 1
            else: # TODO: assignment
                # if nobody is producing I activate the machine less expensive
                machine_production_lst = self.machine_production[:, idx_item]
                setup_costs_lst = self.setup_costs[:, idx_item]
                setup_loss_lst = self.setup_loss[:, idx_item]
                kpi = setup_costs_lst / (machine_production_lst + 0.001)
                # another good policy can be:
                # kpi = setup_costs_lst + 0.0 # + self.env.holding_costs[idx_item] * (machine_production_lst - setup_loss_lst - avg_demand[idx_item])
                # remove machine with 0 production
                kpi[machine_production_lst == 0] = np.inf
                
                for best_machine in np.argsort(kpi):
                    if best_action[best_machine] == 0:
                        best_action[best_machine] = idx_item + 1
                        break
        
        # IF IT IS REASONABLE KEEP PRODUCING:
        for m in range(self.env.n_machines):
            if best_action[m] == 0 and obs['machine_setup'][m] != 0:
                # the machine is producing something
                idx_item = obs['machine_setup'][m] - 1
                # TODO la condizione dovrebbe basarsi su holding cost vs setup cost.
                # if self.runout[idx_item - 1] < self.runout_max:
                inv_if_prod = obs['inventory_level'][idx_item] + self.env.machine_production_matrix[m][idx_item]
                estimated_holding_costs = self.env.holding_costs[idx_item] * inv_if_prod**2 / (2 * avg_demand[idx_item])
                if self.env.setup_costs[m][idx_item] > self.alpha[5] * estimated_holding_costs:
                    best_action[m] = obs['machine_setup'][m]

        return best_action

    def _evaluate_action(self, action, obs):
        pd_machine_setup = [ele for ele in obs['machine_setup']]
        pd_inventory_level = [ele for ele in obs['inventory_level']]
        # EVALUATE
        # print(pd_machine_setup)
        # print(pd_inventory_level)
        setup_costs = np.zeros(self.env.n_machines)
        # Apply action
        self.env._post_decision(
            action,
            pd_machine_setup,
            pd_inventory_level,
            setup_costs
        )
        # tot_cost = sum(setup_costs)
        pd_machine_state = [0] * (1 + self.env.n_items) # 1 is for the iddle state
        for m in range(self.env.n_machines):
            pd_machine_state[pd_machine_setup[m]] += 1

        post_data = []
        post_data.extend(pd_inventory_level)
        post_data.extend(pd_machine_state)
        tot_cost = self.post_decision_value_function.predict(
            np.array(post_data).reshape(1, -1)
        )
        return tot_cost, post_data
    
    def _compute_best_action(self, obs, epsilon_prob=0):
        # self.env.machine_production_matrix
        if self.env.n_machines > 2:
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
        elif self.env.n_machines == 2:
            # ESAUSTIVE SEARCH 2 MACHINES 
            best_val = np.Inf
            for i0 in self.valid_actions[0]:
                for i1 in self.valid_actions[1]:
                    action = [i0, i1]
                    tmp, pd_data = self._evaluate_action(action, obs)    
                    if tmp < best_val:
                        best_val = tmp
                        best_action = action
                        best_post_data = pd_data
        else:
            # EXHAUSTIVE SEARCH FOR 1 MACHINE
            best_action = [0]
            best_val, best_post_data = self._evaluate_action(best_action, obs)
            epsilon = np.random.binomial(1, epsilon_prob)
            if epsilon == 1:
                i = np.random.randint(self.env.n_items + 1)
                best_action = [i]
                if obs['inventory_level'][i-1] + self.env.machine_production_matrix[0][i-1] <= self.env.max_inventory_level[i-1]:
                    best_val, best_post_data = self._evaluate_action(best_action, obs)
            for i in range(1, self.env.n_items + 1): # action 0 has already been analyse before
                action = [i]
                # do not consider actions which lead to a overful inventory
                if obs['inventory_level'][i-1] + self.env.machine_production_matrix[0][i-1] > self.env.max_inventory_level[i-1]:
                    continue
                tmp, pd_data = self._evaluate_action(action, obs)
                if tmp < best_val:
                    best_val = tmp
                    best_action = action
                    best_post_data = pd_data
        return best_val, best_action, best_post_data

    def learn(self, epochs = 10, refit_time=10):
        """
        Implementation of the fixed point itearation
        """
        start_time = time.time()
        N_MC_REP = 4
        # Get initial policy rule
        self.train_policy()
        # penalties = []
        # data = []
        # for i in tqdm(range(epochs)):
        #     # if i % 1000 == 0:
        #     #     self.plot_post_decision_value_function()
        #     # obs = self.env.reset_time()
        #     obs = self.env.reset()
        #     for t in range(self.env.T):
        #         best_val, action, pd_data = self._compute_best_action(
        #             obs,
        #             epsilon_prob = 0.5*(1 - i / epochs)
        #         )

        #         # COMPUTING EXPECTED VALUE ACTUAL COST
        #         temp_dem = self.env.stoch_model.generate_scenario(n_time_steps=100)
        #         tot_lost_sales = 0
        #         tot_holding_costs = 0
        #         tot_future = 0
        #         setup_costs = np.zeros(self.env.n_machines)
        #         machine_setup = [ele for ele in obs['machine_setup']]
        #         original_inventory_level = [ele for ele in obs['inventory_level']]
        #         # already computed in _compute_best_action
        #         self.env._post_decision(
        #             action,
        #             machine_setup,
        #             original_inventory_level,
        #             setup_costs
        #         )

        #         # for ii in range(N_MC_REP):
        #         for d in [[0,0], [1,0], [0,1], [1, 1]]:
        #             lost_sales = np.zeros(self.env.n_items)
        #             holding_costs = np.zeros(self.env.n_items)
        #             inventory_level = [ele for ele in original_inventory_level]
        #             self.env._satisfy_demand(
        #                 lost_sales,
        #                 inventory_level,
        #                 # temp_dem[:,ii],
        #                 d,
        #                 holding_costs
        #             )
        #             tot_lost_sales += sum(lost_sales)
        #             tot_holding_costs += sum(holding_costs)
        #             # se I[0] == 2 -> sono influenzato da uno stato che non vedrò mai.
        #             new_obs = {
        #                 'inventory_level': inventory_level,
        #                 'machine_setup': machine_setup
        #             }

        #             tmp, tmp_action, _ = self._compute_best_action(
        #                 new_obs,
        #                 epsilon_prob=0
        #             )
        #             # print(new_obs)
        #             tot_future += tmp
        #         tot_lost_sales /= N_MC_REP
        #         tot_holding_costs /= N_MC_REP
        #         tot_future /= N_MC_REP
        #         obs, cost, _, info = self.env.step(action, verbose=False)
        #         tot_cost = info['setup_costs'] + tot_lost_sales + tot_holding_costs
        #         data.append(pd_data)
        #         penalties.append(tot_cost + self.discount * tot_future)
        #         self.post_decision_value_function.fit(
        #             pd_data, tot_cost + self.discount * tot_future
        #         )
        #         # simulate_episodes
        #         # self.plot_post_decision_value_function()

        #     # if (i > 0) and (i % refit_time == 0):
        #     #     print("REFIT")
        #     #     X = np.array(data)
        #     #     y = np.array(penalties)
        #     #     self.post_decision_value_function.fit(X,y)
        #     #     self.plot_post_decision_value_function()
        #     #     self.plot_policy()
        #     #     penalties = []
        #     #     data = []
        #     # print("***")
        time_duration = time.time() - start_time
        print(f'Learning time: {time_duration:.2f} s')

    def plot_policy(self, file_path=None): 
        possible_states = self.env.n_items + 1
        cmap = plt.cm.get_cmap('viridis', 3) 
        self.policy = np.zeros(
            shape=(
                possible_states,
                self.env.max_inventory_level[0] + 1,
                self.env.max_inventory_level[1] + 1
            )
        )
        for setup in range(possible_states):
            for inv0 in range(self.env.max_inventory_level[0] + 1):
                for inv1 in range(self.env.max_inventory_level[1] + 1):
                    obs = {
                        'demand': np.array([0, 0]), # does not influence
                        'inventory_level': [inv0, inv1],
                        'machine_setup': [setup]
                    }
                    action = self.get_action(obs)
                    self.policy[setup, inv0, inv1] = action[0]

        fig, axs = plt.subplots(1, possible_states)	
        fig.suptitle('Found Policy')	
        cmap = plt.cm.get_cmap('viridis', 3) 
        for i, ax in enumerate(axs):	
            ax.set_title(f'Setup {i}')	
            im = ax.pcolormesh(
                self.policy[i,:,:],
                cmap = cmap,
                edgecolors='k',
                linewidth=2
            )
            im.set_clim(0, possible_states - 1)	
            if i == 0:	
                ax.set_ylabel('I1')	
            ax.set_xlabel('I2')		
        # COLOR BAR:	
        bound = [0,1,2]
        # Creating 8 Patch instances
        fig.subplots_adjust(bottom=0.2)
        # Creating 8 Patch instances
        ax.legend([mpatches.Patch(color=cmap(b)) for b in bound],
                   ['{}'.format(i) for i in range(3)],
                   loc='upper center', bbox_to_anchor=(-0.8,-0.13),
                   fancybox=True, shadow=True, ncol=3)
        if file_path:
            fig.savefig(file_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()

    def plot_post_decision_value_function(self, file_path=None):
        possible_states = self.env.n_items + 1
        self.post_decision_function = np.zeros(
            shape=(
                possible_states,
                self.env.max_inventory_level[0] + 1,
                self.env.max_inventory_level[1] + 1
            )
        )
        for setup in range(possible_states):
            for inv0 in range(self.env.max_inventory_level[0] + 1):
                for inv1 in range(self.env.max_inventory_level[1] + 1):
                    self.post_decision_function[setup, inv0, inv1] = self.post_decision_value_function.predict(
                        np.array(
                            [
                                inv0, inv1,
                                1 if setup == 0 else 0,
                                1 if setup == 1 else 0,
                                1 if setup == 2 else 0
                            ]
                        ).reshape(1, -1)
                    )
        fig, axs = plt.subplots(nrows=1, ncols=possible_states)	
        fig.suptitle('Post Decision Value Function')	
        for i, ax in enumerate(axs):	
            ax.set_title(f'Setup {i}')	
            im = ax.imshow(	
                self.post_decision_function[i,:,:],	
                aspect='auto', cmap='viridis'	
            )
            for (i, j), z in np.ndenumerate(self.post_decision_function[i,:,:]):
                ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

            ax.invert_yaxis()	
            if i == 0:	
                ax.set_ylabel('I1')	
            ax.set_xlabel('I2')	
        fig.subplots_adjust(right=0.85)	
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])	
        fig.colorbar(im, cax=cbar_ax)
        if file_path:
            fig.savefig(file_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
