# -*- coding: utf-8 -*-
import time
import numpy as np
from numpy import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import product,combinations
from stable_baselines3 import PPO,A2C,DQN
from torch import nn
import torch
import os
from scipy.stats import binom


class MLP(nn.Module):
  '''
    Multilayer Perceptron.
  '''
  def __init__(self,DIR_LOAD,agent_name):
    super().__init__()
    self.agent_name = agent_name
    if agent_name == 'PPO':
        stable_baseline_agent = PPO.load(DIR_LOAD)
        self.extractor = stable_baseline_agent.policy.mlp_extractor.value_net
        self.value_net = stable_baseline_agent.policy.value_net
    if agent_name == 'DQN':
        stable_baseline_agent = DQN.load(DIR_LOAD)
        self.value_net = stable_baseline_agent.q_net.q_net

  def forward(self, x):
    '''Forward pass'''
    if self.agent_name == 'PPO':
        x = self.extractor(x)
    x = self.value_net(x)
    return x


class ApproximateValueIteration():
    """Approximate Value Iteration using Torch as ANN
    """
    def __init__(self, env):
        super(ApproximateValueIteration, self).__init__()
        self.env = env
        # CHECK REQUISITES
        self._check_requisite()
        # INIT VARIABLES
        self.POSSIBLE_STATES = self.env.n_items + 1
        self.model_name = 'PPO'
        BEST_MODEL_DIR = os.path.dirname(os.path.abspath('__file__')) + '/logs' +f'/best_{self.model_name}/best_model'
        self.value_function = MLP(BEST_MODEL_DIR,self.model_name)
        self.policy = np.zeros(shape=(self.POSSIBLE_STATES, self.env.max_inventory_level[0] + 1, self.env.max_inventory_level[1] + 1))
        self.discount = 0.97
        # NB: since we consider MC approach instead of the real experctation, we need to have a huge amount of scenarios.
        self.N_MC_RUN = 40 
        self.n_policy_eval = 30
        
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Use the logs file in the root path of the main.
        self.LOG_DIR = BASE_DIR + '/logs' 

    def get_action(self, obs):
        demand = obs['demand']
        machine_setup = obs['machine_setup']
        inventory_level = obs['inventory_level']
        act = self.policy[machine_setup[0], inventory_level[0], inventory_level[1]]
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

    def learn(self, iterations = 10,use_pretrained_weights = True):
        
        if not use_pretrained_weights:
            for module in self.value_function.children():
                try:
                    for sub_module in module:
                        sub_module.reset_parameters()
                except:
                    try:
                        module.reset_parameters()
                    except: pass
                    
                #module.reset_parameters()
        loss_function = nn.SmoothL1Loss()
        #loss_function = nn.MSELoss()
        
        start_time = time.time()
        optimizer = torch.optim.Adam(self.value_function.parameters(), lr=0.0001)
        eval_hist = []
        pbar = tqdm(range(iterations))
        old_cost_eval = np.Inf
        for epoch in pbar:
            new_values_list = []
            old_values_list = []
            total_loss = 0
            for machine_setup in range(self.POSSIBLE_STATES):
                for inv1 in range(self.env.max_inventory_level[0] + 1):
                    for inv2 in range(self.env.max_inventory_level[1] + 1):
                        tmp_opt = [0] * self.POSSIBLE_STATES
                        tmp_opt_old = [0] * self.POSSIBLE_STATES
                        for action in range(self.POSSIBLE_STATES):
                            # Check feasibility:
                            if action != 0:
                                setup_loss = 0
                                if machine_setup != action and action != 0:
                                    setup_loss = self.env.setup_loss[0][action - 1]
                                production = self.env.machine_production_matrix[0][action - 1] - setup_loss
                                inventory_level = [inv1, inv2]
                                if inventory_level[action - 1] + production > self.env.max_inventory_level[action - 1]:
                                    tmp_opt[action] -= np.Inf
                                if inventory_level[action - 1] + production > self.env.max_inventory_level[action - 1]:
                                    tmp_opt_old[action] -= np.Inf
                                    continue

                            # Otherwise:
                            for i in range(self.N_MC_RUN):
                                demand = self.env.stoch_model.generate_scenario()
                                _ = self.env.reset()
                                # create the inventory list (it must be here otherwise it will update and no recreate)
                                # TODO: this needs to simulate the interaction with the environment in order to optimize the value function
                                inventory_level = [inv1, inv2]
                                self.env.inventory_level = inventory_level.copy()
                                self.env.machine_setup = [machine_setup].copy()
                                self.env.demand = demand
                                obs, reward, done, total_cost = self.env.step([action])
                                cost = reward
                                # total_cost = self.env._take_action(
                                #     [action], [machine_setup], inventory_level, demand
                                # )
                                #cost = sum([ele for key, ele in total_cost.items()])
                                # print(cost, total_cost)
                                # print("***")
                                
                                next_obs = np.expand_dims(np.array([obs['machine_setup'][0], obs['inventory_level'][0], obs['inventory_level'][1]]), axis = 0)
                                next_inv_1 = obs['inventory_level'][0]
                                next_inv_2 = obs['inventory_level'][1]
                                #next_obs = np.expand_dims(np.array([action, inventory_level[0], inventory_level[1]]), axis = 0)
                                next_obs = torch.from_numpy(next_obs).to(torch.float).to(device="cuda")
                                if self.model_name == 'PPO':
                                    next_val = self.value_function.forward(next_obs)[0]
                                if self.model_name == 'DQN':
                                    next_val = self.value_function.forward(next_obs)[0][action]
                                obs = np.expand_dims(np.array([machine_setup, inventory_level[0], inventory_level[1]]), axis = 0)
                                obs = torch.from_numpy(obs).to(torch.float).to(device="cuda")
                                if self.model_name == 'PPO':
                                    present_val = self.value_function.forward(obs)[0]
                                if self.model_name == 'DQN':
                                    present_val = self.value_function.forward(obs)[0][action]
                                tmp_opt_old[action] += present_val 
                                tmp_opt[action] +=  self.discount * next_val - cost
                        # print(machine_setup, inv1, inv2, tmp_opt)
                        # quit()
                        new_value = max(tmp_opt) / self.N_MC_RUN
                        old_value = max(tmp_opt_old) / self.N_MC_RUN
                        # optimizer.zero_grad()
                        # loss = loss_function(old_value,new_value)
                        # loss.backward()         
                        # optimizer.step()
                        # total_loss += loss.item()
                        new_values_list.append(new_value)
                        old_values_list.append(old_value)                        
                        self.policy[machine_setup, inv1, inv2] = tmp_opt.index(max(tmp_opt))
                
            cost_eval = self.eval_policy(self.n_policy_eval)
            if old_cost_eval>cost_eval:
                torch.save(self.value_function,f'{self.LOG_DIR}/best_avi_value_function.pth')
                np.save(f'{self.LOG_DIR}/best_avi_policy.npy',np.array(self.policy))
            eval_hist.append(cost_eval)
            old_values_list = torch.stack(old_values_list)
            new_values_list = torch.stack(new_values_list)
            optimizer.zero_grad()
            loss = loss_function(old_values_list, new_values_list)
            loss.backward()         
            optimizer.step()
            pbar.set_description(f'Loss: {round(loss.item(),0)} | Episode cost eval.: {round(cost_eval,0)}\n')
            
                    
        time_duration = time.time() - start_time
        print(f'\nLearning time: {round(time_duration,2)}s')
        print("\nFinished Learning. \n")
        return eval_hist
        
  
    def load_best_model(self):
        self.value_function = torch.load(f'{self.LOG_DIR}/best_avi_value_function.pth')
        self.policy = np.load(f'{self.LOG_DIR}/best_avi_policy.npy')
        
    def plot_value_function(self,dir_save = 'results', experiment_name = 'VI',seed = 0):
        value_map = np.zeros((self.env.n_items+1,self.env.max_inventory_level[0]+1,self.env.max_inventory_level[1]+1))
        for i in range(self.env.max_inventory_level[0]+1):
            for j in range(self.env.max_inventory_level[1]+1):
                for k in range(self.env.n_items+1):
                    value_list = []
                    for action in range(self.env.n_items+1):
                        obs = np.expand_dims(np.array([k,i,j]), axis = 0)
                        action = np.array([[action]])
                        if torch.cuda.is_available():
                            obs = torch.from_numpy(obs).to(torch.float).to(device="cuda")
                            action = torch.from_numpy(action).to(torch.float).to(device="cuda")
                        else:
                            obs = torch.from_numpy(obs).to(torch.float)
                            action = torch.from_numpy(action).to(torch.float)
                        value = self.value_function(obs)
                        try:value_list.append(value.item())
                        except:value_list.append(value[0][int(action.item())].item())
                    value_map[k,i,j] = np.mean(value_list)
        
        fig, axs = plt.subplots(nrows=1, ncols=self.POSSIBLE_STATES)
        fig.suptitle('Value Function')
        for i, ax in enumerate(axs):
            ax.set_title(f'Stato {i}')
            im = ax.imshow(
                -value_map[i,:,:],
                aspect='auto', cmap='viridis'
            )
            ax.invert_yaxis()
            if i == 0:
                ax.set_ylabel('I1')
            ax.set_xlabel('I2')
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.savefig(f'{dir_save}/value_function_{experiment_name}_{seed}.pdf')
        plt.close()

    def plot_policy(self,dir_save = 'results',experiment_name = 'VI', seed = 0):
        fig, axs = plt.subplots(1, self.POSSIBLE_STATES)
        fig.suptitle('Found Policy')
        for i, ax in enumerate(axs):
            ax.set_title(f'Stato {i}')
            im = ax.pcolormesh(
                self.policy[i,:,:], edgecolors='k', linewidth=2
            )
            im.set_clim(0, self.POSSIBLE_STATES - 1)
            if i == 0:
                ax.set_ylabel('I1')
            ax.set_xlabel('I2')

        # COLOR BAR:
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.savefig(f'{dir_save}/policy_function_{experiment_name}_{seed}.pdf')
        plt.close()

    def plot_hist(self,hist):
        plt.plot(hist)
        plt.title('Historical evaluations')
        plt.show()
        
    def _check_requisite(self):
        if self.env.n_machines > 1:
            raise Exception('ValueIteration is defined for one machine environment')
