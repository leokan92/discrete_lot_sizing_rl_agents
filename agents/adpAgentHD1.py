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



class AdpAgentHD1():
    def __init__(self, env: SimplePlant, settings: dict):
        super(AdpAgentHD1, self).__init__()
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
        action = self._compute_best_action_rule(obs)
        # _, action, _ = self._compute_best_action(obs)
        return action

    def train_policy(self):
        # Same as stochastic gradient
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
            # TODO: speed up considering bisection
            for I in range(self.env.max_inventory_level[i]):
                val = 0
                for d, p in enumerate(empirical_distr[i]):
                    reward = self.env.lost_sales_costs[i] * max(d-I, 0) + self.env.holding_costs[i] * max(I-d, 0)
                    val += p * reward
                if val < best_val:
                    best_val = val
                    best_I = I
            self.best_inv[i] = best_I

        lb = [0, 0, 0, 0, 0]
        ub = [self.env.T, 2, 1, 2, 2]
        alphas = []
        for i1 in range(self.env.T):
            for i2 in range(1, 3):
                for i3 in range(5, 20, 5):
                    for i4 in range(5, 20, 5):
                        for i5 in range(5, 20, 5):
                            alphas.append([i1, i2, i3/10, i4/10, i5/10])

        # from pymoo.algorithms.soo.nonconvex.pso import PSO, PSOAnimation
        # from pymoo.optimize import minimize
        # from pymoo.core.problem import Problem
        # print('\n####################################\n')
        # print('Starting PSO optimization for policy function...\n')
        # start = time.time()
        # lb = [0, 0, 0, 0, 0]
        # ub = [self.env.T, 2, 1, 2, 2]

        # def simulate_run(x):
        #     self.alpha = x
        #     return self.simulate_episodes(use_rule=True)

        # class ProblemWrapper(Problem):
        #     def _evaluate(self, desings, out, *args, **kwargs):
        #         res = []
        #         for desing in desings:
        #             res.append(simulate_run(desing))
        #         out['F'] = np.array(res)
        
        # problem = ProblemWrapper(n_var = len(lb),n_obj = 1, xl = lb, xu = ub)
        # algorithm = PSO(pop_size = 10)

        # res = minimize(
        #     problem,
        #     algorithm,
        #     seed=1,
        #     save_history=True,
        #     verbose=False
        # )
        # print("Best solution found by PSO from pymoo: \nX = %s\nF = %s" % (res.X, res.F))
        # xopt = res.X
        
        # self.w = xopt
        
        # print(f'\nOptimization Finished. Elepsed time: {round(time.time()-start,1)}s')

        # runout time
        # safety per best_inv
        # improve if setup present
        # inventory_level < best_inv
        # if avg_demand > production
        # if setup_cost > estimated_holding_costs
        best_val = np.inf
        for alpha in alphas:
            self.alpha = alpha
            ans = self.simulate_episodes(use_rule=True)
            if ans < best_val:
                best_val = ans
                best_alpha = [ele for ele in alpha]
        print(f"best_val: {best_val}, alpha: {best_alpha}")
        self.alpha = best_alpha
        # self.alpha = [7.14539534, 0.6610792,  0.99876544, 1.198411 ,  0.58298435]
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
            np.where(obs['inventory_level'] < self.alpha[1] * self.best_inv)[0]
        )
        # list_items = np.where(self.runout < self.alpha[0])[0]
        for eligible_item in list_items: #np.where(obs['inventory_level'] < self.best_inv)[0]: #
            priority = self.env.lost_sales_costs[eligible_item] / (self.runout[eligible_item] + 0.01)
            priority += self.alpha[2] * (obs['machine_setup'].count(eligible_item + 1) >= 1)
            # if a machine is producing the item, higher priority 
            # if production is low respect the demand:
            priority += self.alpha[3] * avg_demand[eligible_item] / max(self.machine_production[:, eligible_item])
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
                # the machine is producing something and we have not yet allocate it
                idx_item = obs['machine_setup'][m] - 1
                # se idx_item in ELEGIBLE: continua a produrlo

                inv_if_prod = obs['inventory_level'][idx_item] + self.env.machine_production_matrix[m][idx_item]
                # compute the expect holding cost (it is a triangle with height inv_if_prod and with base: inv_if_prod/avg_demand)
                estimated_holding_costs = self.env.holding_costs[idx_item] * inv_if_prod**2 / (2 * avg_demand[idx_item])
                if self.env.setup_costs[m][idx_item] > self.alpha[4] * estimated_holding_costs:
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
        # Compute exact expected reward
        tot_lost_sales, tot_holding_costs = self._compute_expected_reward(
            pd_inventory_level
        )
        # Get total cost as the sum of the expected reward and future forecast.
        tot_cost = sum(setup_costs) + tot_lost_sales + tot_holding_costs
        tot_cost += self.discount * self.post_decision_value_function.predict(
            np.array(post_data).reshape(1, -1)
        )
        return tot_cost, post_data
    
    def _compute_expected_reward(self, inventory):
        tot_lost_sales = 0
        tot_holding_costs = 0
        for d in [[0,0], [1,0], [0,1], [1, 1]]:
            lost_sales = np.zeros(self.env.n_items)
            holding_costs = np.zeros(self.env.n_items)
            inventory_level = [ele for ele in inventory]
            self.env._satisfy_demand(
                lost_sales,
                inventory_level,
                d,
                holding_costs
            )
            tot_lost_sales += sum(lost_sales)
            tot_holding_costs += sum(holding_costs)
        tot_lost_sales /= 4
        tot_holding_costs /= 4
        return tot_lost_sales, tot_holding_costs

    def _compute_best_action(self, obs, epsilon_prob=0):
        # EXHAUSTIVE SEARCH FOR 1 MACHINE
        best_action = [0]
        best_val, best_post_data = self._evaluate_action(best_action, obs)
        epsilon = np.random.binomial(1, epsilon_prob)
        if epsilon == 1:
            # take a random action
            i = np.random.randint(self.env.n_items + 1)
            best_action = [i]
            if obs['inventory_level'][i-1] + self.env.machine_production_matrix[0][i-1] <= self.env.max_inventory_level[i-1]:
                best_val, best_post_data = self._evaluate_action(best_action, obs)
        if obs['inventory_level'][0] < self.theta[0]:
            best_action = [1]
        elif obs['inventory_level'][1] < self.theta[1]:
            best_action = [2]
        else:
            best_action = [0]
        best_val, best_post_data = self._evaluate_action(best_action, obs)
        # for i in range(1, self.env.n_items + 1): # action 0 has already been analyse before
        #     action = [i]
        #     # do not consider actions which lead to a overful inventory
        #     if obs['inventory_level'][i-1] + self.env.machine_production_matrix[0][i-1] > self.env.max_inventory_level[i-1]:
        #         continue
        #     tmp, pd_data = self._evaluate_action(action, obs)
        #     if tmp < best_val:
        #         best_val = tmp
        #         best_action = action
        #         best_post_data = pd_data
        return best_val, best_action, best_post_data

    def learn(self, epochs = 10):
        self.theta = [1, 1]
        start_time = time.time()
        for i in tqdm(range(epochs)):
            obs = self.env.reset()
            for _ in range(self.env.T):
                _, action, pd_data = self._compute_best_action(
                    obs,
                    epsilon_prob = 0.5 * (1 - i / epochs)
                )
                # compute
                setup_costs = np.zeros(self.env.n_machines)
                machine_setup = [ele for ele in obs['machine_setup']]
                original_inventory_level = [ele for ele in obs['inventory_level']]
                # already computed in _compute_best_action
                self.env._post_decision(
                    action,
                    machine_setup,
                    original_inventory_level,
                    setup_costs
                )
                tot_lost_sales = 0
                tot_holding_costs = 0
                tot_future = 0
                # consider all possible demand to get the exact expected reward
                tot_lost_sales, tot_holding_costs = self._compute_expected_reward(
                   original_inventory_level
                )
                # Next step:
                obs, _, _, _ = self.env.step(action, verbose=False)
                # In Q-leaning we have to take the maximum of the next step (obs)
                tot_future, _, _ = self._compute_best_action(
                    obs,
                    epsilon_prob = 0.5 * (1 - i / epochs)
                )
                tot_cost = sum(setup_costs) + tot_lost_sales + tot_holding_costs
                self.post_decision_value_function.fit(
                    pd_data, tot_cost + self.discount * tot_future
                )
        time_duration = time.time() - start_time
        print(f'Learning time: {time_duration:.2f} s')

    def plot_policy(self, file_path=None): 
        if self.env.n_items != 2 and self.env.n_machines!=1:
            raise ValueError("Not possilbe to print!")
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
        if self.env.n_items != 2 and self.env.n_machines!=1:
            raise ValueError("Not possilbe to print!")
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
