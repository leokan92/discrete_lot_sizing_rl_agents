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
from agents.stochasticProgrammingAgent import StochasticProgrammingAgent
import types
import copy
import gym

class stchasticRl():
    """Value Iteration
    """
    def __init__(self, env,stoch_model):
        super(stchasticRl, self).__init__()
        model_name = 'PPO'
        self.spagent = StochasticProgrammingAgent(env,[2,2], stoch_model)
        self.rlagent = StableBaselineAgent(env,model_name,True)
        self.env = env

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
        if isinstance(obs, dict):
            
            obs = np.concatenate((obs['inventory_level'],obs['machine_setup']),[self.spagent.get_action(obs)])
        act = self.model.predict(obs,deterministic=True)[0]    
        return act
    
    
    
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
        
        
        lb = -np.ones(7+self.env.n_items+1)*3
        ub = np.ones(7+self.env.n_items+1)*3
        
        class ProblemWrapper(Problem):
            def _evaluate(self, desings, out, *args, **kwargs):
                res = []
                for desing in desings:
                    res.append(simulate_run(desing))
                    
                out['F'] = np.array(res)
        
        problem = ProblemWrapper(n_var = len(lb),n_obj = 1, xl = lb, xu = ub)
        
        algorithm = PSO(pop_size = 100)
        
        res = minimize(
            problem,
            algorithm,
            seed=1,
            save_history=True,
            verbose=False
        )
        print("Best solution found by PSO from pymoo: \nX = %s\nF = %s" % (res.X, res.F/iterations))
        xopt = res.X
        print(f'\nOptimization Finished. Elepsed time: {round(time.time()-start,1)}s')

    def save_policy(self,xopt,key):
        #TODO include an save policy (a parameter for the PSO only and integrates with the RL)
        pass
        
    def load_policy(self,key):
        #TODO include an save policy (a parameter for the PSO only and integrates with the RL)
        pass