# -*- coding: utf-8 -*-
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class QLearningAgent():
    """Plain Q learning
    """
    # TODO: generalize to more than 2 items
    def __init__(self, env):
        super(QLearningAgent, self).__init__()
        self.env = env
        # CHECK REQUISITES
        self._check_requisite()
        # INIT VARIABLES
        self.q_matrix = np.zeros(shape=(3, self.env.max_inventory_level[0] + 1, self.env.max_inventory_level[1] + 1, 3))
        self.visit = np.zeros(shape=(3, self.env.max_inventory_level[0] + 1, self.env.max_inventory_level[1] + 1))
        self.epsilon = 0.3
        self.discount = 0.95
        self.alpha = 0.2

    def get_action(self, obs):
        if np.random.uniform() < self.epsilon:
            # print("rnd")
            return self.env.action_space.sample()
        else:
            q_vect = self.q_matrix[self.env.machine_setup[0], self.env.inventory_level[0], self.env.inventory_level[1], :]
            act = np.argmin(q_vect)
            return [act] # metto in lista perché ci potrebbero essere più macchine

    def learn(self, epochs = 1000):
        start_time = time.time()
        # TODO: pass all data from obs
        for epoch in tqdm(range(epochs)):
            done = False
            obs = self.env.reset()
            while not done: # Loop through the episode
                # 1. GET ACTION
                action = self.get_action(obs)
                # print("action:", action)
                # 2. UPDATE 
                self.visit[action, self.env.inventory_level[0], self.env.inventory_level[1]] += 1
                obs, reward, done, info = self.env.step(action)

                q_tilde = reward + self.discount * np.min(
                    self.q_matrix[self.env.machine_setup[0], self.env.inventory_level[0], self.env.inventory_level[1], :]
                )
                self.q_matrix[self.env.machine_setup[0], self.env.inventory_level[0], self.env.inventory_level[1], action[0]] = self.alpha * q_tilde + (1 - self.alpha) * self.q_matrix[self.env.machine_setup[0], self.env.inventory_level[0], self.env.inventory_level[1], action[0]]

        time_duration = time.time() - start_time
        print(time_duration)
        print("\nFinished Learning. \n")


    def compute_value_function_and_policy(self):
        self.value_function = np.zeros(
            shape=(3, self.env.max_inventory_level[0], self.env.max_inventory_level[1])
        )
        self.policy = np.zeros(
            shape=(3, self.env.max_inventory_level[0], self.env.max_inventory_level[1])
        )
        for i in range(3):
            for j in range(self.env.max_inventory_level[0]):
                for k in range(self.env.max_inventory_level[1]):
                    self.policy[i,j,k] = np.argmin(self.q_matrix[i,j,k,:])
                    self.value_function[i,j,k] = np.min(self.q_matrix[i,j,k,:])

    def plot_value_function(self):
        fig, axs = plt.subplots(nrows=1, ncols=3)
        fig.suptitle('Value Function')
        for i, ax in enumerate(axs):
            ax.set_title(f'Stato {i}')
            im = ax.imshow(
                self.q_matrix[i,:,:],
                aspect='auto', cmap='viridis'
            )
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()

    def plot_policy(self):
        fig, axs = plt.subplots(1, 3)
        fig.suptitle('Found Policy')
        for i, ax in enumerate(axs):
            ax.set_title(f'Stato {i}')
            im = ax.imshow(
                self.policy[i,:,:],
                aspect='auto', cmap='viridis'
            )
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        plt.show()
        
    def evaluate_agent(self,epochs = 50,render = False, return_episode_rewards = False):
        rewards = []
        self.epsilon = 0 # Make deterministic actions
        for i in range(0,epochs):
            ep_rewards = []
            done = False
            obs = self.env.reset()
            while not done: # Loop through the episode
                # 1. GET ACTION
                action = self.get_action(obs)
                obs, reward, done, info = self.env.step(action)
                ep_rewards.append(reward)
                if render:
                    self.env.render()
            rewards.append(np.array(ep_rewards))
        rewards = np.stack(rewards)
        if return_episode_rewards:
            return(np.mean(rewards,0),np.std(rewards,0))
        else:
            return(np.mean(rewards),np.std(rewards))  
    

    def save(self):
        pass
    
    def _check_requisite(self):
        if self.env.n_machines > 1:
            raise Exception('QLearningAgent is defined for one machine environment')
