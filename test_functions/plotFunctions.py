# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_policy(agent, file_name=None):
    if agent.env.n_machines > 1:
        print('Impossible to print for n_machine > 1')
        return
    cmap = plt.cm.get_cmap('viridis', 3) 
    policy_map = np.zeros(
        (
            agent.env.max_inventory_level[0]+1,
            agent.env.max_inventory_level[1]+1,
            agent.env.n_items+1
        )
    )
    for i in range(agent.env.max_inventory_level[0]+1):   
        for j in range(agent.env.max_inventory_level[1]+1):
            for k in range(agent.env.n_items+1):
                # TODO: end this general funtion
                obs = np.expand_dims(np.array([i, j, k]), axis = 0)
                action = agent.get_action(obs,deterministic=True)
                policy_map[i,j,k] = action
    agent.policy = policy_map

    fig, axs = plt.subplots(1, agent.POSSIBLE_STATES)
    fig.suptitle('Found Policy')
    for i, ax in enumerate(axs):
        ax.set_title(f'Setup {i}')
        im = ax.pcolormesh(
            agent.policy[:,:,i], cmap = cmap, edgecolors='k', linewidth=2
        )
        im.set_clim(0, agent.POSSIBLE_STATES - 1)
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
    if file_name:
        fig.savefig(
            os.path.join('results', file_name),
            bbox_inches='tight'
        )
    else:
        plt.show()
