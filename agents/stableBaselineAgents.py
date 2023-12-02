# -*- coding: utf-8 -*-
import os
import time
import gym
import torch
import numpy as np
import copy
from envs import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from stable_baselines3 import PPO,A2C,DQN,SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback


class SimplePlantSB(SimplePlant):
    def __init__(self, settings, stoch_model):
        super().__init__(settings, stoch_model)
        try:self.dict_obs = settings['dict_obs']
        except:self.dict_obs = False
        self.last_inventory = copy.copy(self.inventory_level)
        self.action_space = gym.spaces.MultiDiscrete(
            [self.n_items+1] * self.n_machines
        )
        
        if self.dict_obs:
            self.observation_space = gym.spaces.Dict({
                'inventory_level': gym.spaces.Box(low = np.zeros(self.n_items),high = np.ones(self.n_items)*(settings['max_inventory_level'][0]+1)*self.n_items),
                'machine_setup': gym.spaces.MultiDiscrete([self.n_items+1] * self.n_machines),
                'last_inventory_level':gym.spaces.Box(low = np.zeros(self.n_items),high = np.ones(self.n_items)*(settings['max_inventory_level'][0]+1)*self.n_items)
            })
        else:
            self.observation_space = gym.spaces.Box(
                low=np.zeros(2*self.n_items+self.n_machines),# high for the inventory level
                high=np.concatenate(
                    [
                        np.array(self.max_inventory_level),
                        np.ones(self.n_machines) * (self.n_items+1), #high for the machine setups 
                        np.array(self.max_inventory_level) # high for the inventory level
                    ]),
                dtype=np.int32
            )

    def step(self, action):
        """
        Step method: Execute one time step within the environment

        Parameters
        ----------
        action : action given by the agent

        Returns
        -------
        obs : Observation of the state give the method _next_observation
        reward : Cost given by the _reward method
        done : returns True or False given by the _done method
        dict : possible information for control to environment monitoring

        """
        self.last_inventory = copy.copy(self.inventory_level)
            
        self.total_cost = self._take_action(action, self.machine_setup, self.inventory_level, self.demand)
        
        # self.total_cost['setup_costs'] = 0
        # self.total_cost['holding_costs'] = 0
        
        reward = -sum([ele for key, ele in self.total_cost.items()])
        #reward = -self.total_cost['lost_sales']
        
        #reward = np.abs(action)
        
        self.current_step += 1
        done = self.current_step == self.T
        obs = self._next_observation()

        return obs, reward, done, self.total_cost

    def _next_observation(self):
        """
        Returns the next demand
        """
        obs = SimplePlant._next_observation(self)
        obs['last_inventory_level'] = copy.copy(self.last_inventory)
        if isinstance(obs, dict):    
            if not self.dict_obs:
                obs = np.concatenate(
                    (
                        obs['inventory_level'], # n_items size
                        obs['machine_setup'], # n_machine size
                        obs['last_inventory_level']# n_items size
                    )
                )
        else:
            if self.dict_obs:
                raise('Change dict_obst to False')
        return obs


class StableBaselineAgent():
    """
    Stable baseline Agent Agent from StableBaselines3
    We adapt the env to stablebaseline requirements:
    A different _next_observation is required, with the observation space.
    """
    def __init__(self, env: SimplePlant, settings: dict):
        super(StableBaselineAgent, self).__init__()
        
        if settings['multiagent']:
            self.env = env
        else:
            self.env = SimplePlantSB(env.settings, env.stoch_model)
        self.last_inventory = env.inventory_level
        self.model_name = settings['model_name']
        self.experiment_name = settings['experiment_name']
        self.parallelization = settings['parallelization']
        try:self.dict_obs = settings['dict_obs']
        except:self.dict_obs = False
        
        self.POSSIBLE_STATES = self.env.n_items + 1
        self.env.cost_to_reward = True
        self.epsilon = 0

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))    
        # Use the logs file in the root path of the main.
        self.LOG_DIR = os.path.join(BASE_DIR,'logs')

        if self.parallelization:
            # For cpu parallelization in StableBaseline learning
            def make_env(seed):
                def _init():
                    env = self.env
                    env = Monitor(
                        env,
                        os.path.join(f'{self.LOG_DIR}','monitor',f'{self.model_name}_{self.experiment_name}_{seed}'),
                        allow_early_resets=True
                    )
                    return env
                return _init
            num_cpu = 5
            env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        else:
            env = Monitor(
                self.env,
                os.path.join(f'{self.LOG_DIR}','monitor',f'{self.model_name}_{self.experiment_name}')
            )
        self.eval_callback = EvalCallback(
            env,
            best_model_save_path=os.path.join(f'{self.LOG_DIR}',f'best_{self.model_name}_{self.experiment_name}'),
            log_path=f'{self.LOG_DIR}/',
            eval_freq=100,
            deterministic=True,
            verbose=0,
            render=False
        ) 
        if self.dict_obs:
            policy = 'MultiInputPolicy'
        else:
            policy = 'MlpPolicy'
        if self.model_name == 'PPO':
            self.model = PPO(
                policy,
                env,verbose = 0, batch_size = 256, n_steps = 256, gamma = 0.96, gae_lambda = 0.9, n_epochs = 20, ent_coef = 0.0, max_grad_norm = 0.5, vf_coef = 0.5, learning_rate = 5e-3, use_sde = False, clip_range = 0.4, policy_kwargs = dict(log_std_init=-2,ortho_init=False,activation_fn=torch.nn.ReLU,net_arch=[dict(pi=[300, 300], vf=[300, 300])])
            )
        elif self.model_name == 'A2C':
            self.model = A2C(
                policy,
                env,verbose = 0, learning_rate=0.002, n_steps=100, gamma = 0.95, vf_coef = 0.7,policy_kwargs= dict(net_arch=[300, 300]), seed = None
            )
        elif self.model_name == 'DQN':
            self.model = DQN(
                policy,
                env, verbose = 0, learning_rate= 2.3e-3, buffer_size=100000, learning_starts=1000, batch_size=32, tau=1.0, gamma=0.99,target_update_interval=10,train_freq= 256,gradient_steps= 128, exploration_fraction=0.16, exploration_initial_eps=0.04, policy_kwargs= dict(net_arch=[300, 300]), seed = None
            )
        elif self.model_name == 'SAC':
            self.model = SAC(
                policy,
                env, verbose = 0,  learning_rate=0.0003, buffer_size=1000000, learning_starts=1000, batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1,seed = None,action_noise=None, replay_buffer_class=None, replay_buffer_kwargs=None, optimize_memory_usage=False, ent_coef='auto', target_update_interval=1, target_entropy='auto', use_sde=False, sde_sample_freq=-1, use_sde_at_warmup=False, tensorboard_log=None, create_eval_env=False, policy_kwargs=dict(activation_fn=torch.nn.ReLU,net_arch=[dict(pi=[300, 300], vf=[300, 300])])
            )
        
    def get_action(self, obs):
        obs['last_inventory_level'] = copy.copy(self.last_inventory)
        if isinstance(obs, dict):
            if self.dict_obs:
                act = self.model.predict(obs,deterministic=True)[0]
            else:
                list_obs = []
                for item in obs:
                    list_obs.append(obs[item])
                obs_ = np.array(np.concatenate(list_obs))
                act = self.model.predict(obs_,deterministic=True)[0]
        else:
            if self.dict_obs:
                raise('Change the policy to dictionary observations')
            else:
                act = self.model.predict(obs,deterministic=True)[0]
        self.last_inventory = copy.copy(obs['inventory_level'])
        return act

    def learn(self, epochs=1000):
        print(f"{self.model_name} learning...")
        start_time = time.time()

        # We define the EvalCallback wrapper to save the best model            
        # Here the model learns using the provided environment in the Stable baseline Agent definition
        # We mutiply the number of epochs by the number of time periods to give the number of training steps
        self.model.learn(
            epochs*self.env.T,
            callback=self.eval_callback,
            # tb_log_name='PPO'
        )

        self.env.close()

        time_duration = time.time() - start_time
        print(f"Finished Learning {time_duration:.2f} s")
        
    def load_agent(self, path):
        if self.model_name == 'PPO':
            self.model = PPO.load(path)
        elif self.model_name == 'A2C':    
            self.model = A2C.load(path)
        elif self.model_name == 'DQN':    
            self.model = DQN.load(path)
        elif self.model_name == 'SAC':    
            self.model = SAC.load(path)     
        
    def plot_policy(self, seed=1):
        # ONLY WORKING FOR 2 ITEMS 1 MACHINE
        cmap = plt.cm.get_cmap('viridis', 3) 
        policy_map = np.zeros((self.env.max_inventory_level[0]+1,self.env.max_inventory_level[1]+1,self.env.n_items+1))
        for i in range(self.env.max_inventory_level[0]+1):   
            for j in range(self.env.max_inventory_level[1]+1):
                for k in range(self.env.n_items+1):
                    obs = np.expand_dims(np.array([i,j,k]), axis = 0)
                    try: action = self.model.predict(obs,deterministic=True)[0][0][0]
                    except: action = self.model.predict(obs,deterministic=True)[0][0]
                    #print(f'action: {action} | obs: {obs}')
                    policy_map[i,j,k] = action
        self.policy = policy_map
        
        fig, axs = plt.subplots(1, self.POSSIBLE_STATES)
        fig.suptitle('Found Policy')
        for i, ax in enumerate(axs):
            ax.set_title(f'Setup {i}')
            im = ax.pcolormesh(
                self.policy[:,:,i], cmap = cmap, edgecolors='k', linewidth=2
            )
            im.set_clim(0, self.POSSIBLE_STATES - 1)
            ax.set_xlabel('I2')
            if i == 0:
                ax.set_ylabel('I1')

        # COLOR BAR:
        bound = [0,1,2]
        # Creating 8 Patch instances
        fig.subplots_adjust(bottom=0.2)
        ax.legend(
            [mpatches.Patch(color=cmap(b)) for b in bound],
            ['{}'.format(i) for i in range(3)],
            loc='upper center', bbox_to_anchor=(-0.8,-0.13),
            fancybox=True, shadow=True, ncol=3
        )
        fig.savefig(os.path.join(f'results', f'policy_function_{self.model_name}_{self.experiment_name}_{seed}.pdf'), bbox_inches='tight')
        plt.close()
    
    def plot_value_function(self, seed):
        # ONLY WORKING FOR 2 ITEMS 1 MACHINE
        value_map = np.zeros((self.env.max_inventory_level[0]+1,self.env.max_inventory_level[1]+1,self.env.n_items+1))
        for i in range(self.env.max_inventory_level[0]+1):
            for j in range(self.env.max_inventory_level[1]+1):
                for k in range(self.env.n_items+1):
                    value_list = []
                    for action in range(self.env.n_items+1):
                        obs = np.expand_dims(np.array([j,i,k]), axis = 0)
                        action = np.array([[action]])
                        if torch.cuda.is_available():
                            obs = torch.from_numpy(obs).to(torch.float).to(device="cuda")
                            action = torch.from_numpy(action).to(torch.float).to(device="cuda")
                        else:
                            obs = torch.from_numpy(obs).to(torch.float)
                            action = torch.from_numpy(action).to(torch.float)
                        try: 
                            value,prob,dist_entropy = self.model.policy.evaluate_actions(obs,action) 
                            value_list.append(value.item())
                        except: 
                            value = self.model.policy.q_net(obs)[0][int(action.item())]
                            value_list.append(value.item())
                        
                    value_map[j,i,k] = np.array(value_list).mean()     
        
        self.value_function = value_map     
        # Plotting:
        fig, axs = plt.subplots(nrows=1, ncols=self.POSSIBLE_STATES)
        fig.suptitle('Value Function')
        for i, ax in enumerate(axs):
            ax.set_title(f'Setup {i}')
            im = ax.imshow(
                -self.value_function[:,:,i],
                aspect='auto', cmap='viridis'
            )
            if i == 0:
                ax.set_ylabel('I1')
            
            ax.set_xlabel('I2')
            ax.invert_yaxis()
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        
        fig.colorbar(im, cax=cbar_ax)
        fig.savefig(os.path.join('results',f'value_function_{self.model_name}_{self.experiment_name}_{seed}.pdf'))
        plt.close()
