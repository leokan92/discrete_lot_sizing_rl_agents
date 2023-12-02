import os
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom,gamma


def generate_binomial_probs(size = 5,n = 5, p= 0.5,plot = True):
    """
    To print the binomial distribtion vector and plot a discreate binomial distribution
    """
    plt.rcParams.update({'font.family':'times new roman'})

    rv = binom(n, p)

    x = np.arange(size)
    probs = rv.pmf(x)
    diff = 1 - sum(probs)
    diff = diff/len(probs)
    probs = probs + diff
    print('------binomial------')
    print(x)
    print(probs)
    if plot:
        plt.vlines(x, 0, probs, colors='k', linestyles='-', lw=1)
        plt.legend(loc='best', frameon=False)
        plt.savefig(os.path.join('results', 'binomial_dist_2.pdf'))
        plt.show()
        

def plot_ep_evol(folder = 'binomial_3',seed = 0,model_names = ['VI','VIMC','PPO','SP','PSO'],x_type = 'reward',y_label = 'Cost'):    
    plt.rcParams.update({'font.family':'times new roman'})
    fig, axs = plt.subplots(len(model_names),figsize=(20,10))

    i = 0
    for model in model_names:
        x = np.abs(
            np.load(
                    os.path.join(
                        'results',
                        'binomial',
                        f'{model}_{folder}_{x_type}_test_{seed}.npy'
                    )
                )
            )
        axs[i].plot(x[:100],label = model)
        axs[i].set_title(model)
        i += 1

    for ax in axs.flat:
        ax.set(xlabel='Time steps', ylabel=y_label)

    for ax in axs.flat:
        ax.label_outer()    
    plt.savefig(
        os.path.join(
            'results',
            f'evol_reward_{folder}_{seed}.pdf'
        ),
        bbox_inches='tight'
    )
