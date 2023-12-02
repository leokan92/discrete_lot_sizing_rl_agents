# -*- coding: utf-8 -*-
import os
import time
import copy
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class ValueIteration():
    """Value Iteration
    """
    def __init__(self, env, settings):
        super(ValueIteration, self).__init__()
        self.env = copy.copy(env)
        # CHECK REQUISITES
        self._check_requisite()
        # INIT VARIABLES
        self.POSSIBLE_STATES = self.env.n_items + 1
        self.value_function = np.zeros(shape=(self.POSSIBLE_STATES, self.env.max_inventory_level[0] + 1, self.env.max_inventory_level[1] + 1))
        self.policy = np.zeros(shape=(self.POSSIBLE_STATES, self.env.max_inventory_level[0] + 1, self.env.max_inventory_level[1] + 1))
        self.discount = settings['discount_rate']

    def get_action(self, obs):
        machine_setup = obs['machine_setup']
        inventory_level = obs['inventory_level']
        act = self.policy[int(machine_setup[0]), int(inventory_level[0]), int(inventory_level[1])]
        return [int(act)] 
    
    def eval_policy(self,iterations):
        env_val = self.env
        tot_reward = 0
        for i in range(iterations):
            done = False
            obs = env_val.reset()
            while not done:
                action = self.get_action(obs)
                obs, reward, done, info = env_val.step(action)
                tot_reward += reward
        return tot_reward/iterations

    def learn(self, iterations = 10):
        vals = self.env.stoch_model.settings['demand_distribution']['vals']
        probs = self.env.stoch_model.settings['demand_distribution']['probs']
        start_time = time.time()
        for _ in tqdm(range(iterations)):
            for machine_setup in range(self.POSSIBLE_STATES):
                for inv1 in range(self.env.max_inventory_level[0] + 1):
                    for inv2 in range(self.env.max_inventory_level[1] + 1):
                        tmp_opt = [0] * self.POSSIBLE_STATES
                        for action in range(self.POSSIBLE_STATES):
                            # Check feasibility:
                            if action != 0:
                                setup_loss = 0
                                if machine_setup != action and action != 0:
                                    setup_loss = self.env.setup_loss[0][action - 1]
                                production = self.env.machine_production_matrix[0][action - 1] - setup_loss
                                inventory_level = [inv1, inv2]
                                if inventory_level[action - 1] + production > self.env.max_inventory_level[action - 1]:
                                    tmp_opt[action] += np.Inf
                                    continue

                            # Calculates the value for each action considering the probability mass function
                            for demand in itertools.product(vals, vals):
                                # prob_demand = np.prod([probs[i-1] for i in demand]) 
                                # Calculates the probability for a certain demand combitionation
                                prob_demand = np.prod([probs[vals.index(i)] for i in demand])
                                # Creates the variable inventory_level
                                inventory_level = [inv1, inv2]
                                # Calculates the total cost using the method _take_action
                                total_cost = self.env._take_action([action], [machine_setup], inventory_level, demand)
                                # Sums all the costs
                                cost = sum([ele for key, ele in total_cost.items()]) 
                                # The method _take action also modifies the inventory_level
                                next_state_val = self.value_function[action, inventory_level[0], inventory_level[1]]
                                # Sums for a certain action, the cost
                                tmp_opt[action] += prob_demand * (cost + self.discount * next_state_val)
                        if self.value_function[machine_setup, inv1, inv2] != min(tmp_opt):
                            # Considering all the actions we get the value with the lower cost
                            self.value_function[machine_setup, inv1, inv2] = min(tmp_opt)
                            # We use the index of the lower cost as the best action for the policy
                            self.policy[machine_setup, inv1, inv2] = tmp_opt.index(min(tmp_opt))
                            reach_convergence = False
                        else:
                            reach_convergence = True
            if reach_convergence:
                print("reached convergence")
                break
        time_duration = time.time() - start_time
        print(f'\nLearning time: {round(time_duration,2)}s')
        print("\nFinished Learning. \n")

    def save_model(self,seed = 42,experiment_name = ''):
        np.save(os.path.join('logs',f'value_function_{experiment_name}_vi_{seed}.npy'),self.value_function)
        np.save(os.path.join('logs',f'policy_function_{experiment_name}_vi_{seed}.npy'),self.policy)
    
    def load_model(self,seed = 42,experiment_name = ''):
        self.value_function = np.load(os.path.join('logs',f'value_function_{experiment_name}_vi_{seed}.npy'),allow_pickle=True)
        self.policy = np.load(os.path.join('logs',f'policy_function_{experiment_name}_vi_{seed}.npy'),allow_pickle=True)


    def plot_value_function(self, file_path=None):
        fig, axs = plt.subplots(nrows=1, ncols=self.POSSIBLE_STATES)	
        fig.suptitle('Value Function')	
        for i, ax in enumerate(axs):	
            ax.set_title(f'Setup {i}')	
            im = ax.imshow(	
                self.value_function[i,:,:],	
                aspect='auto', cmap='viridis'	
            )
            for (i, j), z in np.ndenumerate(self.value_function[i,:,:]):
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

    def plot_policy(self, file_path=None):
        # fig, axs = plt.subplots(1, self.POSSIBLE_STATES)
        fig, axs = plt.subplots(1, self.POSSIBLE_STATES, figsize=(30, 10))
        cmap = plt.cm.get_cmap('viridis', 3) 
        for i, ax in enumerate(axs):	
            ax.set_title(f'Setup {i}', fontsize="30")	
            im = ax.pcolormesh(self.policy[i,:,:],cmap = cmap, edgecolors='k', linewidth=2)	
            im.set_clim(0, self.POSSIBLE_STATES - 1)	
            if i == 0:	
                ax.set_ylabel('I1', size=26,)	
            ax.set_xlabel('I2', size=26,)		
            ax.tick_params(axis='both', which='major', labelsize=26)
        # COLOR BAR:	
        bound = [0,1,2]
        # Creating 8 Patch instances
        fig.subplots_adjust(bottom=0.2)
        # Creating 8 Patch instances
        ax.legend([mpatches.Patch(color=cmap(b)) for b in bound],
                   ['{}'.format(i) for i in range(3)],
                   loc='upper center', bbox_to_anchor=(-0.8,-0.13),
                   fancybox=True, shadow=True, ncol=3, fontsize="26")
        if file_path:
            fig.savefig(file_path, bbox_inches='tight')
        else:
            plt.show()
        plt.close()
    	
    def _check_requisite(self):	
        if self.env.n_machines > 1:	
            raise Exception('ValueIteration is defined for one machine environment')
        if self.env.n_items != 2:
            raise Exception('ValueIteration is defined for two items environment')
        if self.env.stoch_model.settings['demand_distribution']['name'] != 'probability_mass_function':	
            raise Exception('ValueIteration is only available for probability_mass_function distributions')

'''
NB: considering 1 machine the assignment is satisfied.
'''