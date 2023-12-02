# -*- coding: utf-8 -*-
import time
import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

class ValueIterationMC():
    """Value Iteration
    """
    # TODO: generalize to more than 2 items
    def __init__(self, env):
        super(ValueIterationMC, self).__init__()
        self.env = env
        # CHECK REQUISITES
        self._check_requisite()
        # INIT VARIABLES
        self.POSSIBLE_STATES = self.env.n_items + 1
        self.value_function = np.zeros(shape=(self.POSSIBLE_STATES, self.env.max_inventory_level[0] + 1, self.env.max_inventory_level[1] + 1))
        self.policy = np.zeros(shape=(self.POSSIBLE_STATES, self.env.max_inventory_level[0] + 1, self.env.max_inventory_level[1] + 1))
        self.discount = 0.9
        # NB: since we consider MC approach instead of the real experctation, we need to have a huge amount of scenarios.
        self.N_MC_RUN = 20 
        self.n_policy_eval = 10

    def get_action(self, obs):
        demand = obs['demand']
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
        # learn mc means learn with Monte Carlo instead of exact vi
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

                            # Otherwise:
                            for i in range(self.N_MC_RUN):
                                demand = self.env.stoch_model.generate_scenario()
                                # create the inventory list (it must be here otherwise it will update and no recreate)
                                inventory_level = [inv1, inv2]
                                total_cost = self.env._take_action(
                                    [action], [machine_setup], inventory_level, demand
                                )
                                cost = sum([ele for key, ele in total_cost.items()])
                                next_val = self.value_function[action, inventory_level[0], inventory_level[1]]
                                tmp_opt[action] += cost + self.discount * next_val
                        self.value_function[machine_setup, inv1, inv2] = min(tmp_opt) / self.N_MC_RUN
                        self.policy[machine_setup, inv1, inv2] = tmp_opt.index(min(tmp_opt))

        time_duration = time.time() - start_time
        print(f'\nLearning time: {round(time_duration,2)}s')
        print("\nFinished Learning. \n")

    def save_model(self,seed = 42, experiment_name = ''):
        np.save(os.path.join('logs',f'value_function_vimc_{experiment_name}_{seed}.npy'),self.value_function)
        np.save(os.path.join('logs',f'policy_function_vimc_{experiment_name}_{seed}.npy'),self.policy)
    
    def load_model(self,seed = 42, experiment_name = ''):
        self.value_function = np.load(os.path.join('logs',f'value_function_vimc_{experiment_name}_{seed}.npy'))
        self.policy = np.load(os.path.join('logs',f'policy_function_vimc_{experiment_name}_{seed}.npy'))

    def plot_value_function(self,experiment_name='VIMC',dir_save='',seed = 42):	
        fig, axs = plt.subplots(nrows=1, ncols=self.POSSIBLE_STATES)	
        fig.suptitle('Value Function')	
        for i, ax in enumerate(axs):	
            ax.set_title(f'Setup {i}')	
            im = ax.imshow(	
                self.value_function[i,:,:],	
                aspect='auto', cmap='viridis'	
            )	
            ax.invert_yaxis()	
            if i == 0:	
                ax.set_ylabel('I1')	
            ax.set_xlabel('I2')	
        fig.subplots_adjust(right=0.85)	
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])	
        fig.colorbar(im, cax=cbar_ax)	
        fig.savefig(os.path.join(f'{dir_save}',f'value_function_{experiment_name}_{seed}.pdf'))
        plt.close()

    def plot_policy(self, experiment_name = 'VIMC', dir_save = '', seed = 42):	
        fig, axs = plt.subplots(1, self.POSSIBLE_STATES)	
        fig.suptitle('Found Policy')	
        cmap = plt.cm.get_cmap('viridis', 3) 
        for i, ax in enumerate(axs):	
            ax.set_title(f'Setup {i}')	
            im = ax.pcolormesh(self.policy[i,:,:],cmap = cmap, edgecolors='k', linewidth=2)	
            im.set_clim(0, self.POSSIBLE_STATES - 1)	
            if i == 0:	
                ax.set_ylabel('I1')	
            ax.set_xlabel('I2')	
        bound = [0,1,2]
        fig.subplots_adjust(bottom=0.2)
        # Creating 8 Patch instances
        ax.legend([mpatches.Patch(color=cmap(b)) for b in bound],
                    ['{}'.format(i) for i in range(3)],
                    loc='upper center', bbox_to_anchor=(-0.8,-0.13),
                    fancybox=True, shadow=True, ncol=3)
        fig.savefig(os.path.join(f'{dir_save}',f'policy_function_{experiment_name}_{seed}.pdf'), bbox_inches='tight')
        plt.close()

    def _check_requisite(self):	
        if self.env.n_machines > 1:	
            raise Exception('ValueIteration is defined for one machine environment')
        if self.env.n_items != 2:
            raise Exception('ValueIteration is defined for two items environment')
        if self.env.stoch_model.settings['demand_distribution']['name'] != 'probability_mass_function':	
            raise Exception('ValueIteration for probability_mass_function distributions')

'''
NB: considering 1 machine the assignment is satisfied.
'''