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
import os
from pymoo.algorithms.soo.nonconvex.pso import PSO, PSOAnimation
from pymoo.optimize import minimize
from pymoo.core.problem import Problem
from agents.stableBaselineAgents import StableBaselineAgent

class RlPSO():
    """Value Iteration
    """
    def __init__(self, env,experiment_name = '',rl_model_name = 'PPO'):
        super(RlPSO, self).__init__()
        model_name = 'PPO'
        self.rlagent = StableBaselineAgent(env,model_name,True)
        BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{rl_model_name}_{experiment_name}/best_model')
        #BEST_MODEL_DIR = os.path.dirname(os.path.abspath('__file__')) + '/logs' +f'/best_{model_name}/best_model'
        self.rlagent.load_agent(BEST_MODEL_DIR)
        self.env = env
        # CHECK REQUISITES
        #
        # INIT VARIABLES
        self.POSSIBLE_STATES = self.env.n_items + 1
        self.value_function = np.zeros(shape=(self.POSSIBLE_STATES, self.env.max_inventory_level[0] + 1, self.env.max_inventory_level[1] + 1))
        self.policy = np.zeros(shape=(self.POSSIBLE_STATES, self.env.max_inventory_level[0] + 1, self.env.max_inventory_level[1] + 1))
        self.n_parameters = 5
        self.w = np.ones(self.n_parameters)
        
        # For binomial distribution we use the following exp_demand:
        # We use a probability mass function in the smaller complexity scenario and the binomial distribution expected demand in higher demand scenario
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
        priority = np.zeros(self.env.n_items)
        action = []
        rl_action = self.rlagent.get_action([*machine_setup,*inventory_level])
        for j in range(self.env.n_machines):
            priority = np.ones(self.env.n_items)*rl_action[j]*self.w[0] + np.ones(self.env.n_items)*self.w[1] + np.ones(self.env.n_items)*machine_setup[j]*self.w[2] + inventory_level*self.w[4] + self.exp_demand*self.w[3]
            act = np.argmax(priority)
            action.append(act)
        return action
    
    def learn(self, iterations = 10):
        
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
        
        
        lb = -np.ones(self.n_parameters)*1
        ub = np.ones(self.n_parameters)*1
        
        class ProblemWrapper(Problem):
            def _evaluate(self, desings, out, *args, **kwargs):
                res = []
                for desing in desings:
                    res.append(simulate_run(desing))
                out['F'] = np.array(res)
        
        problem = ProblemWrapper(n_var = len(lb),n_obj = 1, xl = lb, xu = ub)
        
        algorithm = PSO(pop_size = 60)
        
        
        res = minimize(problem,
                        algorithm,
                        seed=1,
                        save_history=True,
                        verbose=False)
        print("Best solution found by PSO from pymoo: \nX = %s\nF = %s" % (res.X, res.F/iterations))
        xopt = res.X
        
        #xopt = [0.8599,0.3322,2.0897]
        
    
        print(f'\nOptimization Finished. Elepsed time: {round(time.time()-start,1)}s')
        return xopt
    
    def save_policy(self,xopt,seed = 42,experiment_name = ''):
        np.save(os.path.join('logs',f'policy_opt_rlpso_{seed}_{experiment_name}.npy'),np.array(xopt))
        
    def load_policy(self,seed = 42,experiment_name = ''):
        xopt = np.load(os.path.join('logs',f'policy_opt_rlpso_{seed}_{experiment_name}.npy'))
        self.w = xopt