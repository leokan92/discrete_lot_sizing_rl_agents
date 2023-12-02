# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 23:33:59 2022

@author: leona
"""
import os
import gc
import json
import random
import numpy as np
from envs import *
from agents import *
import pandas as pd
import seaborn as sns
from test_functions import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scenarioManager.stochasticDemandModel import StochasticDemandModel
from test_functions.tablePlotting import table_plotting

sns.set_style({'font.family':'serif', 'font.serif':['Times New Roman']})
plt.rcParams['font.family'] = 'Times New Roman'

# loading the file:
def create_df_result(models,experiment,variable_name,execution_type,cum_sum):
    path = os.path.dirname(os.path.abspath('__file__'))    
    experiment = experiment
    execution_type = execution_type
    variable_file_name = variable_name.lower().replace(' ','_')
    variable_name = variable_name
    df_results = pd.DataFrame(columns= ['Models','N',variable_name])
    T = 0
    for model in models:
        model = model
        result = os.path.join(path,'results',model+'_'+experiment+'_'+variable_file_name+'_'+execution_type+'.npy')
        variable = np.load(result).T
        T = len(variable)
        if cum_sum:
            cummulative_variable = np.cumsum(variable,0)# Cost cummulative sum:
        else:        
            cummulative_variable = variable
        for i in range(len(variable)):
            df_temp = pd.DataFrame({'Models':[model]*len(cummulative_variable[i]),'Time steps':[i]*len(cummulative_variable[i]),variable_name:cummulative_variable[i]})
            df_results = df_results.append(df_temp)
    df_results = df_results.reset_index(drop=True)
    return df_results, T

def plot_variable(models,variables,experiment_name,execution_type,cum_sum):
    for variable_name in variables:
        df_results,T = create_df_result(models,experiment_name,variable_name,execution_type,cum_sum)
        df_results = df_results.rename(columns={'Lost Sales': 'Lost Sales Cost'}) 
        if variable_name == 'Lost Sales':
            variable_name = 'Lost Sales Cost'
        df_results = df_results.replace('RP','MS')
        f = plt.figure(figsize=(10,5))
        g = sns.lineplot(data=df_results, x="Time steps", y=variable_name, hue='Models',style='Models',palette = 'binary',ci=95)
        g.set_xticks(np.arange(0, T, 10))
        if cum_sum:
            file_name = experiment_name+'_'+variable_name.lower().replace(' ','_')+'_'+execution_type+'_sum.pdf'
        else:
            file_name = experiment_name+'_'+variable_name.lower().replace(' ','_')+'_'+execution_type+'.pdf'
        path = os.path.dirname(os.path.abspath('__file__'))   
        path_output = result = os.path.join(path,'results',file_name)
        f.savefig(path_output, bbox_inches='tight')


def plot_policy(agent,env,model_name,experiment_name):
    # ONLY WORKING FOR 2 ITEMS 1 MACHINE
    print(f'generating {model_name} plot...')
    cmap = plt.cm.get_cmap('viridis', 3) 
    policy_map = np.zeros((env.max_inventory_level[0]+1,env.max_inventory_level[1]+1,env.n_items+1))
    obs = {}
    for i in range(env.max_inventory_level[0]+1):   
        for j in range(env.max_inventory_level[1]+1):
            for k in range(env.n_items+1):
                inventory = [i,j]
                setup = [k]
                obs['inventory_level'] = inventory
                obs['machine_setup'] = setup
                action = agent.get_action(obs)
                policy_map[i,j,k] = np.array(action)
    policy = policy_map
    
    fig, axs = plt.subplots(1, env.n_items+1)
    #fig.suptitle('Found Policy')
    for i, ax in enumerate(axs):
        ax.set_title(f'Setup {i}')
        im = ax.pcolormesh(
            policy[:,:,i], cmap = cmap, edgecolors='k', linewidth=2
        )
        im.set_clim(0, env.n_items+1 - 1)
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
    
    fig.savefig(os.path.join(f'results', f'policy_function_{model_name}_{experiment_name}.pdf'), bbox_inches='tight')
    plt.close()
        
# ADPHS_4items_2machines_ADHPS_1_holding_costs_test.npy
# ADPHS_4items_2machines_ADHPS_1_holding_costs_test.npy

execution_type = 'test'
experiment_names = ['20items_10machines']#,'15items_5machines'
models = ['ADPHS','PSO','MultiAgent_ADPHS_PPO']       
variables = ['Holding Costs', 'Lost Sales','Setup Costs']

# plotting the graphs:

# experiment_names = ['15items_5machines_t100']
# models = ['ADPHS3','PSO','MultiAgent_ADPHS_PPO']   
# for experiment_name in experiment_names:
#     plot_variable(models,variables,experiment_name,execution_type,cum_sum = False)

# # plotting the graphs:
# for experiment_name in experiment_names:
#     plot_variable(models,variables,experiment_name,execution_type,cum_sum = True)

# experiment_names = ['4items_2machines','10items_5machines','15items_5machines']
# for experiment_name in experiment_names:
#     plot_variable(models,variables,experiment_name,execution_type,cum_sum = False)

# Plotting tables:
# experiment_names = ['15items_5machines']
# '4items_2machines'

# models = ['PI','VI','ADPHS','PPO','RP','PSO'] 
# table_plotting(model_names = models, experiment_names = experiment_names, execution_type = execution_type,PI_proportion = True)   
 
    
experiment_names = ['2items_VI_binomial_3']
models = ['PI','VI','ADPHS','ADPHS3','A2C','PPO','RP','PSO']  
table_plotting(model_names = models, experiment_names = experiment_names, execution_type = execution_type,PI_proportion = True)
    
experiment_names = ['4items_2machines','10items_5machines','15items_5machines']
models = ['PI','ADPHS','ADPHS3','A2C','PPO','RP','PSO','MultiAgent_ADPHS_PPO']  
table_plotting(model_names = models, experiment_names = experiment_names, execution_type = execution_type,PI_proportion = True)   

experiment_names = ['15items_5machines_t100']
models = ['ADPHS','ADPHS3','A2C','PPO','PSO','MultiAgent_ADPHS_PPO']   
table_plotting(model_names = models, experiment_names = experiment_names, execution_type = execution_type,PI_proportion = False)   

experiment_names = ['15items_5machines_t100','15items_5machines_i100']
models = ['ADPHS','ADPHS3','A2C','PPO','PSO','MultiAgent_ADPHS_PPO']   
table_plotting(model_names = models, experiment_names = experiment_names, execution_type = execution_type,PI_proportion = False)   

experiment_names = ['25items_10machines']
models = ['ADPHS','A2C','PPO','PSO','MultiAgent_ADPHS_PPO']  
table_plotting(model_names = models, experiment_names = experiment_names, execution_type = execution_type,PI_proportion = False)  

# experiment_names = ['100items_15machines']
# models = ['ADPHS','PPO','A2C','MultiAgent_ADHPS_PPO','Ensemble']  
# table_plotting(model_names = models, experiment_names = experiment_names, execution_type = execution_type,PI_proportion = False)  

# experiment_names = ['15items_5machines_t100']
# models = ['PI','ADPHS','PPO','RP','PSO']  
# table_plotting(model_names = models, experiment_names = experiment_names, execution_type = execution_type,PI_proportion = True)   



# Plotting Graphs from values lits:
    
# inputs
# time = np.array([1.9, 2.9, 4.4, 10.7, 22.9,30.5])
# test = np.array([5, 10, 20, 40, 80,160])

df = pd.read_csv('results_time_MS.csv').astype('int')

# plot using lineplot
f = plt.figure(figsize=(10,5))
sns.lineplot(data=df,x='N', y='Time', hue = 'M')
file_name = 'time_processing_machine.pdf'
path = os.path.dirname(os.path.abspath('__file__'))   
path_output = result = os.path.join(path,'results',file_name)
f.savefig(path_output, bbox_inches='tight')


###################################################################
# Plotting 2 items 1 machine policies
###################################################################


# Setting the seeds
np.random.seed(1)
random.seed(10)
# Environment setup load:
experiment_name = '2items_VI_binomial_3' # we set the experiment using the available files in cfg
fp = open(f"./cfg_env/setting_{experiment_name}.json", 'r')
settings = json.load(fp)
fp.close()

# Models setups:
stoch_model = StochasticDemandModel(settings)
settings['time_horizon'] = 20
env = SimplePlant(settings, stoch_model)
settings['dict_obs'] = False
setting_sol_method = {
    'discount_rate': 0.9,
    'experiment_name': experiment_name,
    'parallelization': False,
    'model_name': 'A2C',
    'multiagent':False,
    'branching_factors': [4, 2, 2],
    'dict_obs': False # To be employed if dictionary observations are necessary
}
# Parameters for the ADPHS:
setting_sol_method['regressor_name'] = 'plain_matrix_I2xM1'
setting_sol_method['discount_rate'] = 0.9
agents = []
# Parameters for the RL:
   
training_epochs_RL = 50000
training_epochs_multiagent = 5000

setting_sol_method['parallelization'] = False # Employed to accelerate the RL training (5 jobs in paralle - can be changged in the Stablebaseline agent directly)
env = SimplePlant(settings, stoch_model)

# Number of test execution (number of complet environment iterations)
nreps = 100

#########################################################################
# RL A2C
env = SimplePlant(settings, stoch_model)
setting_sol_method['model_name'] = 'A2C'
base_model_name = 'A2C'
a2c_agent = StableBaselineAgent(
    env,
    setting_sol_method
)

#load best agent before appending in the test list
BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
a2c_agent.load_agent(BEST_MODEL_DIR)
agents.append(("A2C", a2c_agent))

#########################################################################
# RL PPO


setting_sol_method['model_name'] = 'PPO'
base_model_name = 'PPO'
ppo_agent = StableBaselineAgent(
    env,
    setting_sol_method
)

#load best agent before appending in the test list
BEST_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath('__file__')),'logs',f'best_{base_model_name}_{experiment_name}','best_model')
ppo_agent.load_agent(BEST_MODEL_DIR) # For training purposes
agents.append(("PPO", ppo_agent))


###########################################################################
#PSO

pso_agent = PSOagent(env)

pso_agent.load(experiment_name = experiment_name)
agents.append(
    ("PSO", pso_agent)
)
###########################################################################
# VI

vi_agent = ValueIteration(env, setting_sol_method)
vi_agent.learn(iterations=1000)
agents.append(
    ("VI", vi_agent)
)
###########################################################################
# RP
stoch_agent = StochasticProgrammingAgent(
    env,
    setting_sol_method
)
agents.append(("MS", stoch_agent))
###########################################################################
# ADPHS
setting_sol_method['dict_obs'] = False
settings['dict_obs'] = False
setting_sol_method['parallelization'] = True
env = SimplePlant(settings, stoch_model)
adphd_agent = AdpAgentHD(env, setting_sol_method)
adphd_agent.learn(epochs=1000) # does not have load method yet
agents.append(
    ("ADPHS", adphd_agent)
)

#########################################################################
#plotting:
for key,agent in agents:
    plot_policy(agent,env,model_name = key,experiment_name = experiment_name) 

