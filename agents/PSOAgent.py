# -*- coding: utf-8 -*-
import time
import numpy as np
from tqdm import tqdm
from pyswarm import pso
from numpy import random
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.patches as mpatches
import os
from pymoo.algorithms.soo.nonconvex.pso import PSO, PSOAnimation
from pymoo.optimize import minimize
from pymoo.core.problem import Problem


class PSOagent():
    def __init__(self, env):
        super(PSOagent, self).__init__()
        self.env = env
        # CHECK REQUISITES

        # INIT VARIABLES
        self.POSSIBLE_STATES = self.env.n_items + 1
        self.value_function = np.zeros(shape=(self.POSSIBLE_STATES, self.env.max_inventory_level[0] + 1, self.env.max_inventory_level[1] + 1))
        self.policy = np.zeros(shape=(self.POSSIBLE_STATES, self.env.max_inventory_level[0] + 1, self.env.max_inventory_level[1] + 1))
        
        # Policy parameters:
        # TODO EDO: why 6?
        self.w = np.ones(6*self.env.n_machines)
        
        # ValueTree parameters:
        self.discount = 0.8
        self.valuetree = MLPRegressor(solver='sgd', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
        self.learning_rate = 0.2
        
        self.num_iter = 150
        self.num_steps = 5 # Number of steps ahead to check the value of an action
        self.num_scenarios = 5 # Number of scenarios to calculate the mean
        self.num_poststates = 600
        
        # We use a probability mass function in the smaller complexity scenario and the binomial distribution expected demand in higher demand scenario
        # TODO EDO: 1. use an if 2, why the need to exp_demand exact?
        try:
            vals = self.env.stoch_model.settings['demand_distribution']['vals']
            probs = self.env.stoch_model.settings['demand_distribution']['probs']
            self.exp_demand = self.env.n_items*[sum(np.array(vals) *np.array(probs))]
            
        except:
        # For binomial distribution we use the following exp_demand:
            n = self.env.stoch_model.settings['demand_distribution']['n']
            p = self.env.stoch_model.settings['demand_distribution']['p']
            self.exp_demand = np.ones(self.env.n_items)*p*n # We define based on the p = 0.7, n = 1

        self.N_MC_RUN = 100
        
    
    def get_action(self, obs):     
        machine_setup = obs['machine_setup']
        inventory_level = obs['inventory_level']
        priority = np.zeros(self.env.n_items+1)
        action = []
        for j in range(self.env.n_machines):
            for i in range(self.env.n_items):
                if machine_setup[j] != i+1:
                    setup_change = 1
                else:
                    setup_change = 0
                #TODO: needs to be generalized for multiple machines
                priority[i+1] = np.tanh((inventory_level[i]*self.w[(j+1)*5-5] - self.exp_demand[i]-setup_change*self.env.setup_loss[j][i]*self.w[(j+1)*5-4] + (self.env.machine_production_matrix[j][machine_setup[j]-1])*self.w[(j+1)*5-3])*self.env.lost_sales_costs[i]*self.w[(j+1)*5-0] + inventory_level[i]*self.env.holding_costs[i]*self.w[(j+1)*5-1] + setup_change*self.env.setup_costs[j][i]*self.w[(j+1)*5-2])
            act = np.argmax(priority)
            action.append(act)
        return action


    def learn(self, iterations = 10,seed = 42, experiment_name = ''):
        

        def simulate_run(x): # Simulates the an episode iteration
            tot_reward = 0
            self.w = x
            for i in range(iterations):
                done = False
                obs = self.env.reset()
                while not done:
                    action = self.get_action(obs)
                    #print(action)
                    obs, reward, done, info = self.env.step(action)
                    #print(f'Reward: {reward} | Obs: {obs} | Last Act: {action}')
                    tot_reward += reward
            return tot_reward
        
        #################################################################################
        # Learns the policy
        #################################################################################
        
           
        print('\n####################################\n')
        print('Starting PSO optimization for policy function...\n')
        start = time.time()
        
        
        lb = -np.ones(6*self.env.n_machines)*3
        ub = np.ones(6*self.env.n_machines)*3
        
        class ProblemWrapper(Problem):
            def _evaluate(self, desings, out, *args, **kwargs):
                res = []
                for desing in desings:
                    res.append(simulate_run(desing))
                    
                out['F'] = np.array(res)
        
        problem = ProblemWrapper(n_var = len(lb),n_obj = 1, xl = lb, xu = ub)
        
        algorithm = PSO(pop_size = 100)
        
        
        res = minimize(problem,
                        algorithm,
                        seed=1,
                        save_history=True,
                        verbose=False)
        print("Best solution found by PSO from pymoo: \nX = %s\nF = %s" % (res.X, res.F/iterations))
        xopt = res.X
        
        self.w = xopt
        self.save_policy(xopt,experiment_name = experiment_name)
        
        print(f'\nOptimization Finished. Elepsed time: {round(time.time()-start,1)}s')
            
    def plot_value_function(self,seed = 42,dir_save = 'results',experiment_name = ''):
        for machine_setup in range(self.POSSIBLE_STATES):
            for inv1 in range(self.env.max_inventory_level[0] + 1):
                for inv2 in range(self.env.max_inventory_level[1] + 1):
                    obs = np.concatenate((np.array([inv1, inv2]),np.array([machine_setup])))
                    self.value_function[machine_setup, inv1, inv2] = self.valuetree.predict([obs])[0]

        fig, axs = plt.subplots(1, self.POSSIBLE_STATES)

        fig.suptitle('Found Value Function')
        for i, ax in enumerate(axs):
            ax.set_title(f'Setup {i}')
            im = ax.pcolormesh(
                self.value_function[i,:,:]
            )
            #im.set_clim(0, self.POSSIBLE_STATES - 1)
            if i == 0:
                ax.set_ylabel('I1')
            
            ax.set_xlabel('I2')
            #print(self.value_function[i,:,:])

        # COLOR BAR:
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.savefig(os.path.join(f'{dir_save}',f'value_function_VT_{experiment_name}_{seed}.png'))
        plt.close()

    def plot_policy(self,seed = 42,dir_save = 'results',experiment_name = ''):
        cmap = plt.cm.get_cmap('viridis', 3) 
        for machine_setup in range(self.POSSIBLE_STATES):
            for inv1 in range(self.env.max_inventory_level[0] + 1):
                for inv2 in range(self.env.max_inventory_level[1] + 1):
                    obs = {
                        'machine_setup': [machine_setup],
                        'inventory_level': [inv1, inv2]
                    }
                    self.policy[machine_setup, inv1, inv2] = self.get_action(obs)[0]

        fig, axs = plt.subplots(1, self.POSSIBLE_STATES)
        fig.suptitle('Found Policy')
        for i, ax in enumerate(axs):
            ax.set_title(f'Setup {i}')
            im = ax.pcolormesh(
                self.policy[i,:,:],cmap=cmap, edgecolors='k', linewidth=2
            )
            im.set_clim(0, self.POSSIBLE_STATES - 1)
            if i == 0:
                ax.set_ylabel('I1')
            
            ax.set_xlabel('I2')

        # COLOR BAR:
        bound = [0,1,2]
        # Creating 8 Patch instances
        fig.subplots_adjust(bottom=0.2)
        ax.legend(
            [mpatches.Patch(color=cmap(b)) for b in bound],
            ['{}'.format(i) for i in range(3)],
            loc='upper center',
            bbox_to_anchor=(-0.8,-0.13),
            fancybox=True, shadow=True, ncol=3
        )
        fig.savefig(os.path.join(f'{dir_save}',f'policy_function_PSO_{experiment_name}_{seed}.pdf'), bbox_inches='tight')
        plt.close()

    def save_policy(self,xopt,seed = 42,experiment_name = ''):
        np.save(os.path.join('logs',f'policy_opt_pso_{seed}_{experiment_name}.npy'),np.array(xopt))
        
    def load(self,seed = 42,experiment_name = ''):
        xopt = np.load(os.path.join('logs',f'policy_opt_pso_{seed}_{experiment_name}.npy'))
        self.w = xopt

    def _check_requisite(self):
        if self.env.n_machines > 1:
            raise Exception('ValueIteration is defined for one machine environment')
