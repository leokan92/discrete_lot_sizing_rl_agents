# -*- coding: utf-8 -*-
import json
import random
import numpy as np
from envs import *
from agents import ValueIteration
# from test_functions import *
from scenarioManager.stochasticDemandModel import StochasticDemandModel


experiment_name = '2items_VI_binomial_1'
# experiment_name = '2items_VI_binomial_2'

fp = open(f"./cfg_env/setting_{experiment_name}.json", 'r')
settings = json.load(fp)
fp.close()
settings["time_horizon"] = 10

stoch_model = StochasticDemandModel(settings)

env = SimplePlant(settings, stoch_model)

fp = open(f"./cfg_sol/sol_setting.json", 'r')
setting_sol_method = json.load(fp)
fp.close()
agents = []

# VI
vi_agent = ValueIteration(env, setting_sol_method)
vi_agent.learn(iterations=1000)
agents.append(
    ("VI", vi_agent)
)
vi_agent.plot_policy('/home/edo/Pictures/DLSP_Leo/PI_VI_1')
# vi_agent.plot_policy('/home/edo/Pictures/DLSP_Leo/PI_VI_2')

"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

file_path = '/home/edo/Pictures/DLSP_Leo/PI_VI_2'
fig, axs = plt.subplots(1, vi_agent.POSSIBLE_STATES, figsize=(30, 10))
cmap = plt.cm.get_cmap('viridis', 3) 
for i, ax in enumerate(axs):	
    ax.set_title(f'Setup {i}', fontsize="30")	
    im = ax.pcolormesh(vi_agent.policy[i,:,:],cmap = cmap, edgecolors='k', linewidth=2)	
    im.set_clim(0, vi_agent.POSSIBLE_STATES - 1)	
    if i == 0:	
        ax.set_ylabel('I1', size=26,)	
    ax.set_xlabel('I2', size=26,)		
    ax.tick_params(axis='both', which='major', labelsize=26)
# COLOR BAR:	
bound = [0,1,2]
# Creating 8 Patch instances
fig.subplots_adjust(bottom=0.2)
# Creating 8 Patch instances
ax.legend([mpatches.Patch(color=cmap(b)) for b in bound],
            ['{}'.format(i) for i in range(3)],
            loc='upper center', bbox_to_anchor=(-0.8,-0.13),
            fancybox=True, shadow=True, ncol=3, fontsize="26")
fig.savefig(file_path, bbox_inches='tight')
plt.close()
"""