# -*- coding: utf-8 -*-
import os
import gym
import numpy as np
from envs import *
import copy
from agents.stableBaselineAgents import StableBaselineAgent
from scenarioManager.stochasticDemandModel import StochasticDemandModel
  
class SimplePlantSB1(SimplePlant):
    def __init__(self, settings, stoch_model, action_position,base_rl_agent):
        super().__init__(settings, stoch_model)
        self.base_rl_agent = base_rl_agent
        self.action_position = action_position
        self.next_action = np.zeros(self.n_machines)
        self.dict_obs = settings['dict_obs']
        self.action_space = gym.spaces.Discrete(
            self.n_items+1
        )
        
        if self.dict_obs:
            self.observation_space = gym.spaces.Dict({
                'inventory_level': gym.spaces.Box(low = np.zeros(self.n_items),high = np.ones(self.n_items)*(settings['max_inventory_level'][0]+1)*self.n_items),
                'machine_setup': gym.spaces.MultiDiscrete([self.n_items+1] * self.n_machines),
                'next_action': gym.spaces.MultiDiscrete([self.n_items+1] * self.n_machines)
            })
        else:    
            self.observation_space = gym.spaces.Box(
                low=np.zeros(self.n_items+self.n_machines+self.n_machines),
                high=np.concatenate(
                    [
                        np.array(self.max_inventory_level), # high for the inventory level
                        np.ones(self.n_machines) * (self.n_items+1), # high for the machine setups 
                        np.ones(self.n_machines) * (self.n_items+1)  # high for the machine setups (PPO agent best)
                    ]),
                dtype=np.int32
            )
            
            

    def _take_action(self, action, machine_setup, inventory_level, demand):

        self.production = 0
        self.total_cost = np.array([0,0,0,0,0])
        setup_costs = np.zeros(self.n_machines)
        setup_loss = np.zeros(self.n_machines, dtype=int)
        lost_sales = np.zeros(self.n_items)
        holding_costs = np.zeros(self.n_items)

        # if we are just changing the setup, we use the setup cost matrix with the corresponding position given by the actual setup and the new setup
        m = self.action_position 
        if action[m] != 0: # if the machine is not iddle
            # 1. IF NEEDED CHANGE SETUP
            if machine_setup[m] != action[m] and action[m] != 0:
                setup_costs[m] = self.setup_costs[m][action[m] - 1] 
                setup_loss[m] = self.setup_loss[m][action[m] - 1]
            machine_setup[m] = action[m]
            # 2. PRODUCTION
            self.production = self.machine_production_matrix[m][action[m] - 1] - setup_loss[m]
            inventory_level[action[m] - 1] += self.production
            if inventory_level[action[m] - 1] > self.max_inventory_level[action[m] - 1]:
                inventory_level[action[m] - 1] = self.max_inventory_level[action[m] - 1]
        else:
            machine_setup[m] = 0


        # 3. SATIFING DEMAND
        for i in range(0, self.n_items):
            inventory_level[i] -= demand[i]
            if inventory_level[i] < 0:
                lost_sales[i] = - inventory_level[i] * self.lost_sales_costs[i]
                inventory_level[i] = 0
            # 4. HOLDING COSTS
            holding_costs[i] += inventory_level[i] * self.holding_costs[i]
        
        return {
            f'setup_costs_{self.action_position}': sum(setup_costs),
            f'lost_sales_{self.action_position}': sum(lost_sales),
            f'holding_costs_{self.action_position}': sum(holding_costs),
        }
    
    
    def step(self, action):

        self.next_action[self.action_position] = copy.copy(action)  # we use the base agent action for the other setups.
        
        self.next_action = list(map(int, self.next_action))
        #self.next_action = self.next_action.astype(int) #_take_action requires int format 
        
        self.total_cost = self._take_action(self.next_action, self.machine_setup, self.inventory_level, self.demand)
        
        reward = -sum([ele for key, ele in self.total_cost.items()])
        
        self.current_step += 1
        done = self.current_step == self.T
        
        obs = self._next_observation()
        
        
        self.next_action = self.base_rl_agent.get_action(self.obs_sb) # The observation is different because the _next_observation returns the observation with the other agents actions
        #self.next_action = self.base_rl_agent.get_action(self.obs_sim)
        
        obs = self._next_observation() # to give the correct next action of the base agent
        
        return obs, reward, done, self.total_cost
    
    def _next_observation(self):
        """
        Returns the next demand
        """
        obs = SimplePlant._next_observation(self)
        
        # We need another observation for the base agent that does not include the other agents actions
        # self.obs_sb = np.concatenate((np.array(obs['inventory_level']), # n_items size
        #                               np.array(obs['machine_setup'])))
        self.obs_sb = copy.copy(obs)
        
        if self.dict_obs:
            obs['next_action'] = copy.copy(list(self.next_action))
        else:
            obs = np.concatenate((np.array(obs['inventory_level']), # n_items size
                                  np.array(obs['machine_setup']),
                                  copy.copy(self.next_action)))
        
        return obs

class MultiAgentRL():
    def __init__(self, settings, stoch_model, setting_sol_method, base_rl_agent=None):
        super(MultiAgentRL, self).__init__()
        if not base_rl_agent:
            raise "Plase include base agent"
        self.setting_sol_method = setting_sol_method
        self.base_rl_agent = base_rl_agent
        self.parallelization = setting_sol_method['parallelization']
        self.stoch_model = stoch_model
        self.settings = settings
        self.rl_model_name = setting_sol_method['model_name']
        self.experiment_name = setting_sol_method['experiment_name']
        self.last_action = np.zeros(self.settings['n_machines'])
        self.dict_obs = setting_sol_method['dict_obs']
        
    def learn(self, epochs = 1000):
        print("MultiAgentRL learning...")
        # learning phase for the multi-agent:
        self.rl_agents = []
        for i in range(self.settings['n_machines']):
            experiment_name_i = f'{self.experiment_name}_{self.base_rl_agent.model_name}_{i}'
            #experiment_name_i = f'{self.experiment_name}_SP_{i}'
            self.setting_sol_method['experiment_name'] = experiment_name_i
            self.env_mod = SimplePlantSB1(
                self.settings,
                self.stoch_model,
                action_position = i,
                base_rl_agent = self.base_rl_agent
            )
            rl_agent = StableBaselineAgent(
                self.env_mod,
                self.setting_sol_method
            )
            
            # Transfer learning between agents:
            ###############################################################
            # This way it is not working. Maybe it is better to transfer only the model or the neural net (maybe the value function)
            # env = rl_agent.model.env
            # if i > 0:
            #     experiment_name_i = f'{self.experiment_name}_{self.base_rl_agent.model_name}_0' # uses the first agent to transfer learning
            #     rl_agent.load_agent(
            #         os.path.join(f'{rl_agent.LOG_DIR}',f'best_{self.rl_model_name}_'+experiment_name_i,
            #                       'best_model')
            #     )
            #     rl_agent.model.env = env
            #     rl_agent.learn(epochs=epochs)
                
            # else:
            #     rl_agent.learn(epochs=epochs*4) # Each ep with 200 steps
                
            ###############################################################    
                
            rl_agent.learn(epochs = epochs)    
            self.rl_agents.append(rl_agent)
                             

    def load_agent(self):
        self.rl_agents = []
        for i in range(self.settings['n_machines']):
            experiment_name_i = f'{self.experiment_name}_{self.base_rl_agent.model_name}_{i}'
            #experiment_name_i = f'{self.experiment_name}_SP_{i}'
            self.setting_sol_method['experiment_name'] = experiment_name_i
            self.env_mod = SimplePlantSB1(self.settings,
                                          self.stoch_model,
                                          action_position = i,
                                          base_rl_agent = self.base_rl_agent)
            rl_agent = StableBaselineAgent(
                self.env_mod,
                self.setting_sol_method
                )
            self.rl_agents.append(rl_agent)
            self.rl_agents[i].load_agent(
                os.path.join(f'{self.rl_agents[i].LOG_DIR}',f'best_{self.rl_model_name}_'+experiment_name_i,
                             'best_model')
            )
            
    def get_action(self, obs):
        
        base_agent_action = self.base_rl_agent.get_action(obs)
        obs['next_action'] = copy.copy(list(base_agent_action))
        if not self.dict_obs:
            list_obs = []
            for item in obs:
                list_obs.append(obs[item])
            obs = np.array(np.concatenate(list_obs))
        
        
        all_agents_action = base_agent_action*1
        multi_agent_actions = []

        
        for i in range(self.settings['n_machines']):

            action = self.rl_agents[i].get_action(obs)

            if self.dict_obs:
                obs['next_action'] = all_agents_action
            else:    
                obs[-self.settings['n_machines']:] = all_agents_action
            
            multi_agent_actions.append(action)
            
        action = np.array(multi_agent_actions)
        
        if self.dict_obs:
            obs['next_action'] = action
        else:
            obs[-self.settings['n_machines']:] = action
        
        
        # To use the best action based on the action of other agents
        multi_agent_actions = []
        for i in range(self.settings['n_machines']):
            action = self.rl_agents[i].get_action(obs)
            all_agents_action[i] = action
            if self.dict_obs:
                obs['next_action'] = all_agents_action
            else:
                obs[-self.settings['n_machines']:] = all_agents_action
            multi_agent_actions.append(action)
        action = np.array(multi_agent_actions).T

        return action
