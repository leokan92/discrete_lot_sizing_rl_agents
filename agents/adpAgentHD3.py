# -*- coding: utf-8 -*-
import os
import copy
import time
import math
import logging
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



class AdpAgentHD3():
    def __init__(self, env: SimplePlant, settings: dict):
        super(AdpAgentHD3, self).__init__()
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
        # action = self._compute_best_action_rule(obs)
        _, _, action, _ = self._compute_best_action(obs)
        return action

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
    
    def _evaluate_action(self, action, obs):
        pd_machine_setup = [ele for ele in obs['machine_setup']]
        pd_inventory_level = [int(ele) for ele in obs['inventory_level']]
        # Simulate action and get post decision state
        setup_costs = np.zeros(self.env.n_machines)
        self.env._post_decision(
            action,
            pd_machine_setup,
            pd_inventory_level,
            setup_costs
        )
        pd_machine_state = [0] * (self.env.n_items + 1) #  + 1 is for the iddle state
        for m in range(self.env.n_machines):
            pd_machine_state[pd_machine_setup[m]] += 1
        # compute actual expected reward (holding and lost estimated after production)
        tot_lost_sales, tot_holding_costs = self._compute_expected_reward(
            pd_inventory_level
        )
        reward = sum(setup_costs) + tot_lost_sales + tot_holding_costs
        # estimate future value
        post_data = []
        post_data.extend(pd_inventory_level)
        post_data.extend(pd_machine_state)
        future = self.post_decision_value_function.predict(
            np.array(post_data).reshape(1, -1)
        )
        return reward, future, post_data
    
    def _train_expected_reward(self):
        n_scenarios = 10000
        scenario_demand = self.env.stoch_model.generate_scenario(
            n_time_steps=n_scenarios
        )
        self.estimated_lost_sales = []
        self.estimated_holding_costs = []
        for idx_item in range(self.env.n_items):
            self.estimated_lost_sales.append(
                np.zeros(self.env.max_inventory_level[idx_item] + 1)
            )
            self.estimated_holding_costs.append(
                np.zeros(self.env.max_inventory_level[idx_item] + 1)
            )
            for inv_level in range(0, self.env.max_inventory_level[idx_item] + 1):
                self.estimated_lost_sales[idx_item][inv_level] = sum(self.env.lost_sales_costs[idx_item] * np.maximum(
                    scenario_demand[idx_item,:] - inv_level, 0
                )) / n_scenarios
                self.estimated_holding_costs[idx_item][inv_level] = sum(self.env.holding_costs[idx_item] * np.maximum(
                    inv_level - scenario_demand[idx_item,:], 0
                )) / n_scenarios
        
    def _compute_expected_reward(self, inventory):
        tot_lost_sales = 0
        tot_holding_costs = 0
        for i, ele in enumerate(inventory):
            tot_lost_sales += self.estimated_lost_sales[i][ele]
            tot_holding_costs += self.estimated_holding_costs[i][ele]
        return tot_lost_sales, tot_holding_costs

    def _compute_best_action(self, obs, epsilon_prob=0):
        # NB: we should minimize over the post decision.
        # Instead we minimize over reward + gamma Future
        best_reward, best_future, best_action, best_post_data = None,None,None,None
        if self.env.n_machines == 1:
            # EXHAUSTIVE SEARCH FOR 1 MACHINE
            best_action = [0]
            best_reward, best_future, best_post_data = self._evaluate_action(best_action, obs)
            best_val = best_reward + self.discount * best_future
            epsilon = np.random.binomial(1, epsilon_prob)
            if epsilon == 1:
                logging.info("rnd action")
                # choose a random action
                i = np.random.randint(self.env.n_items + 1)
                # if action is feasible
                if obs['inventory_level'][i-1] + self.env.machine_production_matrix[0][i-1] <= self.env.max_inventory_level[i-1]:
                    best_action = [i]
                    best_reward, best_val, best_post_data = self._evaluate_action(best_action, obs)
                    return best_reward, best_val, best_action, best_post_data
            else:   
                # choose the best action        
                for i in range(1, self.env.n_items + 1): # action 0 has already been analyse before
                    action = [i]
                    # do not consider actions which lead to a overful inventory
                    if obs['inventory_level'][i-1] + self.env.machine_production_matrix[0][i-1] > self.env.max_inventory_level[i-1]:
                        continue
                    reward, future, pd_data = self._evaluate_action(action, obs)
                    val = reward + self.discount * future
                    if val < best_val:
                        best_val = val
                        best_future = future
                        best_reward = reward
                        best_action = action
                        best_post_data = pd_data
        elif self.env.n_machines == 2:
            best_val = np.inf
            for i1 in range(self.env.n_items + 1):
                for i2 in range(self.env.n_items + 1):
                    # TODO: farlo diventare vettoriale
                    condition = True
                    if i1 != 0:
                        # production > 0
                        condition &= self.env.machine_production_matrix[0][i1 - 1] != 0
                        # possible to produce
                        prod1 = self.env.machine_production_matrix[0][i1 - 1]
                        if i1 != obs['machine_setup'][0]:
                            prod1 -= self.env.setup_loss[0][i1 - 1]
                        condition &= obs['inventory_level'][i1 - 1] + prod1 < self.env.max_inventory_level[i1 - 1]
                    if i2 != 0:
                        # production > 0
                        condition &= self.env.machine_production_matrix[1][i2 - 1] != 0
                        # possible to produce
                        prod2 = self.env.machine_production_matrix[1][i2 - 1]
                        if i2 != obs['machine_setup'][1]:
                            prod2 -= self.env.setup_loss[1][i2 - 1]
                        condition &= obs['inventory_level'][i2 - 1] + prod2 < self.env.max_inventory_level[i2 - 1]
                    if i1 == i2 and i1 != 0:
                        condition &= obs['inventory_level'][i2 - 1] + prod1 + prod2 < self.env.max_inventory_level[i1 - 1]
                    
                    if  condition:
                        action = [i1, i2] 
                        reward, val, post_data = self._evaluate_action(action, obs)
                        # print(action, reward, val)
                        if reward + self.discount * val < best_val:
                            best_val = reward + self.discount * val
                            best_reward = reward
                            best_future = val
                            best_action = action
                            best_post_data = post_data
        else:
            def generate_successors(node):
                machine = len(node[0])
                if machine == self.env.n_machines:
                    return []
                else:
                    ans = []
                    for ele in self.env.possible_machine_production[machine]:
                        new_inventory = update_inventory(node[2], machine, ele)
                        # holding_cost = np.dot(self.env.holding_costs, new_inventory)
                        # but taking the estimated holding cost can be nice 
                        if all(new_inventory[i] < self.env.max_inventory_level[i] for i in range(self.env.n_items) ):
                            holding_cost = 0
                            for i, inv in enumerate(new_inventory):
                                holding_cost += self.estimated_holding_costs[i][inv]
                        else:
                            holding_cost = np.inf
                            # con questo check qui, quello sotto dovrebbe essere inutile.
                        ans.append(
                            (
                                node[0] + [ele],
                                node[1] + get_setup_costs(machine, ele) + holding_cost,
                                # node[1] + get_setup_costs(machine, ele),
                                new_inventory
                            )
                        )
                    return ans

            def get_setup_costs(m, s):
                if s != 0 and s != obs['machine_setup'][m]:
                    return self.env.setup_costs[m][s - 1]
                return 0

            def update_inventory(inventory_level, m, state):
                tmp = [ele for ele in inventory_level]
                if state != 0: # if the machine is not iddle
                    if obs['machine_setup'][m] != state:
                        production = self.env.machine_production_matrix[m][state - 1] - self.env.setup_loss[m][state - 1]
                    else:
                        production = self.env.machine_production_matrix[m][state - 1]
                    tmp[state - 1] += production
                return tmp

            best_val = np.inf
            nodes = []
            nodes = generate_successors([[], 0, obs['inventory_level']])
            while len(nodes) > 0:
                node_to_examine = nodes.pop(0)
                if len(node_to_examine[0]) == self.env.n_machines:
                    action = node_to_examine[0]
                    reward, val, post_data = self._evaluate_action(action, obs)
                    if reward + self.discount * val < best_val:
                        best_val = reward + self.discount * val
                        best_reward = reward
                        best_future = val
                        best_action = action
                        best_post_data = post_data
                else:
                    new_nodes = generate_successors(node_to_examine)
                    for ele in new_nodes:
                        # Check if node is feasible, else remove it here
                        if ele[1] > best_val:
                            # print("Opt Cut")
                            continue
                        if all(ele[2][i] < self.env.max_inventory_level[i] for i in range(self.env.n_items) ):
                            nodes.insert(0, ele)
                        else:
                            continue
                            # print("Feasibility Cut")
        # print(best_action)
        return best_reward, best_future, best_action, best_post_data

    def learn(self, epochs = 10):
        self._train_expected_reward()
        start_time = time.time()
        for i in tqdm(range(epochs)):
            obs = self.env.reset() # maybe add the initial inventory
            # self.env.inventory_level = [0, 0]
            logging.info("# ################## #")
            logging.info(f"# ### Epoch: {i} ### #")
            logging.info("# ################## #")
            for t in range(self.env.T):
                logging.info(f"**** [epoch: {i}]t: {t} ****")
                logging.info(f"obs: {obs}")
                actual, _, action, pd_data = self._compute_best_action(
                    obs,
                    epsilon_prob = 0.5 * (1 - i / epochs)
                )
                logging.info(f"act: {action}")
                # Next step:
                obs, _, _, _ = self.env.step(action, verbose=False)
                # In Q-leaning we have to take the maximum of the next step (obs)
                # in this way we minimize r + gamma V and not V
                _, tot_future, _, _ = self._compute_best_action(
                    obs
                )
                logging.info(f"\t new val; {actual:.2f} + {self.discount:.2f} * {tot_future:.2f}")
                logging.info(f"\t New record: pd_data: {pd_data} | val: {actual + self.discount * tot_future}")
                self.post_decision_value_function.fit(
                    pd_data, actual + self.discount * tot_future
                )
                logging.info(f"\t V[0] {[f'{ele:.2f}' for ele in self.post_decision_value_function.V[0]]}")
                logging.info(f"\t V[1] {[f'{ele:.2f}' for ele in self.post_decision_value_function.V[1]]}")
                logging.info(f"\t V[2] {[f'{ele:.2f}' for ele in self.post_decision_value_function.V[2]]}")
                logging.info(f"\t V[3] {[f'{ele:.2f}' for ele in self.post_decision_value_function.V[3]]}")
        time_duration = time.time() - start_time
        print(f'Learning time: {time_duration:.2f} s')

    def plot_policy(self, file_path=None): 
        if self.env.n_items != 2 and self.env.n_machines!=1:
            raise ValueError("Not possible to print!")
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
