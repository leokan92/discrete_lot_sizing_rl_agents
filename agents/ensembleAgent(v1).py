# -*- coding: utf-8 -*-
import os
import gym
import numpy as np
from envs import *
import copy
from agents.stableBaselineAgents import StableBaselineAgent
from scenarioManager.stochasticDemandModel import StochasticDemandModel
 
def change_order(last_action,action):
    possible_positions = np.arange(len(action))
    new_action = []
    for i in range(len(last_action)):
        if action[i]<len(possible_positions):
            new_action.append(last_action[possible_positions[action[i]]])
            possible_positions = np.delete(possible_positions,action[i])
        else:
            new_action.append(last_action[possible_positions[0]])
            possible_positions = np.delete(possible_positions,0)

    return new_action

def cross_entropy(predictions, targets, epsilon=1e-10):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce_loss = -np.sum(np.sum(targets * np.log(predictions + 1e-5)))/N
    return ce_loss    

class SimplePlantSB1(SimplePlant):
    def __init__(self, settings, stoch_model,first_agent,second_agent):
        super().__init__(settings, stoch_model)
        self.first_agent = first_agent
        self.second_agent = second_agent
        self.dict_obs = settings['dict_obs']
        self.last_action = np.array(settings['initial_setup'])
        #self.action_space = gym.spaces.Box(low = np.zeros(self.n_machines),high = np.ones(self.n_machines)*1) # depending o
        self.action_space = gym.spaces.Discrete(2)
        # self.action_space = gym.spaces.MultiDiscrete(
        #     [self.n_items+1] * self.n_machines
        # )
        # action_space_shape = []
        # for i in range(self.n_items+1):
        #     action_space_shape.append([(self.n_items+1-i)])
        
        # self.action_space = gym.spaces.MultiDiscrete(
        #     [self.n_machines]*(self.n_machines)
        # )
        
        # self.observation_space = gym.spaces.MultiDiscrete(
        #     [self.n_items+1] * self.n_machines
        # )
        
        # self.action_space = gym.spaces.MultiDiscrete(
        #     [self.n_items+1] * self.n_machines
        # )
        
        if self.dict_obs:
            self.observation_space = gym.spaces.Dict({
                'inventory_level': gym.spaces.Box(low = np.zeros(self.n_items),high = np.ones(self.n_items)*(settings['max_inventory_level'][0]+1)*self.n_items),
                'machine_setup': gym.spaces.MultiDiscrete([self.n_items+1] * self.n_machines),
                'agent_action': gym.spaces.MultiDiscrete([self.n_items+1] * self.n_machines)
            })
        else:
            self.observation_space = gym.spaces.Box(
                low=np.zeros(self.n_items+self.n_machines),
                high=np.concatenate(
                    [
                        np.array(self.max_inventory_level), # high for the inventory level
                        np.ones(self.n_machines) * (self.n_items+1), #high for the machine setups 
                        np.ones(self.n_machines) * (self.n_items+1) #high for the machine setups 
                    ]),
                dtype=np.int32
            )        
        
    def step(self, action):
        """
        Step method: Execute one time step within the environment

        """
        
        obs = self._next_observation()
        # teacher_action = self.second_agent.get_action(self.obs_dict)
        
        if action == 1:  
            action = self.second_agent.get_action(obs)
        else:
            action = copy.copy(self.last_action)
        self.last_action = copy.copy(action)
        # reward = -cross_entropy(action,teacher_action)
        #reward = np.sum(-np.abs(action - teacher_action))
        # self.last_action = self.second_agent.get_action(self.obs_dict)
        
        # action = change_order(self.last_action,action)

        # if action > 0:  
        #     action = self.second_agent.get_action(self.obs_dict)
        # else:
        #     action = copy.copy(self.last_action)
        # self.last_action = copy.copy(action)
        # if action > 0:  
        #     action = self.first_agent.get_action(obs)
        # else:
        #     action = self.second_agent.get_action(self.obs_dict)
        
        # action_1 = self.first_agent.get_action(obs)
        # action_2 = self.second_agent.get_action(self.obs_dict)
        
        # action = np.around(action,0)
        # action_1 = action_1*action
        # action_2 = action_2*(np.ones(len(action)) - action)
        # action = action_1 + action_2
        
        # action = action.astype(int)
        self.total_cost = self._take_action(action, self.machine_setup, self.inventory_level, self.demand)
        
        reward = -sum([ele for key, ele in self.total_cost.items()])
        
        self.current_step += 1
        done = self.current_step == self.T
        obs = self._next_observation()

        return obs, reward, done, self.total_cost

    def _next_observation(self):
        
        """
        Returns the next demand
        """
        obs = SimplePlant._next_observation(self)
        self.obs_dict = copy.copy(obs) # change the order agent
        # obs = self.last_action # change the order agent
        obs['agent_action'] = self.last_action
        if isinstance(obs, dict):    
            if not self.dict_obs:
                obs = np.concatenate(
                    (
                        obs['inventory_level'], # n_items size
                        obs['machine_setup'], # n_machine size
                        obs['agent_action']
                    )
                )
        else:
            if self.dict_obs:
                raise('Change dict_obst to False')
        return obs
        
        
        return obs

class EnsembleAgent():
    def __init__(self, settings, stoch_model, setting_sol_method, first_agent,second_agent):
        super(EnsembleAgent, self).__init__()
        self.setting_sol_method = setting_sol_method
        self.first_agent = first_agent
        self.second_agent = second_agent
        self.stoch_model = stoch_model
        self.settings = settings
        self.model_name = 'Ensemble'
        self.rl_model_name = setting_sol_method['model_name']
        self.experiment_name = setting_sol_method['experiment_name']
        self.dict_obs = settings['dict_obs']
        self.last_action = np.array(settings['initial_setup'])
        
        self.env_mod = SimplePlantSB1(
            self.settings,
            self.stoch_model,
            first_agent = self.first_agent,
            second_agent = self.second_agent
        )
        experiment_name = f'{self.experiment_name}_ensemble_{self.second_agent.model_name}'
        self.setting_sol_method['experiment_name'] = experiment_name
        self.setting_sol_method['dict_obs'] = self.dict_obs
        self.ensemble_agent = StableBaselineAgent(
            self.env_mod,
            self.setting_sol_method
        )
        
        
    def learn(self, epochs = 1000):
        
        self.ensemble_agent.learn(epochs = epochs)       

    def load_agent(self):
        single_agent_name = self.setting_sol_method['model_name']
        self.ensemble_agent.load_agent(
             os.path.join(
                f'{self.ensemble_agent.LOG_DIR}',
                f'best_{single_agent_name}_{self.experiment_name}_ensemble_{self.second_agent.model_name}',
                'best_model'
                )
            )
 
    def get_action(self, obs):
        
        # self.last_action = self.second_agent.get_action(obs)
        
        # obs = self.last_action
        
        obs['agent_action'] = self.last_action
        
        # if isinstance(obs, dict):
        #     if self.dict_obs:
        #         action = self.ensemble_agent.get_action(obs)[0]
        #     else:
        #         list_obs = []
        #         for item in obs:
        #             list_obs.append(obs[item])
        #         obs_ = np.array(np.concatenate(list_obs))
        #         action = self.ensemble_agent.get_action(obs_)[0]
        # else:
        #     if self.dict_obs:
        #         raise('Change the policy to dictionary observations')
        #     else:
        #         action = self.ensemble_agent.get_action(obs)[0]
        
        
        
        
        action = self.ensemble_agent.get_action(obs)
        
        #action = change_order(self.last_action,action)
        
        # action = action + self.second_agent.get_action(obs)
        
        if action == 1:  
            action = self.second_agent.get_action(obs)
        else:
            action = copy.copy(self.last_action)
        self.last_action = copy.copy(action)
        # if action > 0:  
        #     action = self.first_agent.get_action(obs)
        # else:
        #     action = self.second_agent.get_action(obs)
        # action_1 = self.first_agent.get_action(obs)
        # action_2 = self.second_agent.get_action(obs)
        
        # action = np.around(action,0)
        # action_1 = action_1*action
        # action_2 = action_2*(np.ones(len(action)) - action)
        # action = action_1 + action_2
        # action = action.astype(int)        
        return action
