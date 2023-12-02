# -*- coding: utf-8 -*-
import os
from envs import *
import numpy as np
from scenarioManager import *
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from models.multistageOptimization import MultistageOptimization


class StochasticProgrammingAgent():

    def __init__(self, env: SimplePlant, settings: dict):
        super(StochasticProgrammingAgent, self).__init__()
        self.env = env
        self.stoch_model = env.stoch_model
        self.branching_factors = settings['branching_factors']
        self.time_horizon = len(self.branching_factors)
        # NB: since the process is stationary,
        # time_steps can be confused with scenario realization
        selected_data = self.stoch_model.generate_scenario(
            n_time_steps=100
        )
        FFreducer = Fast_forward_W2(selected_data)
        # # NB: questo metodo sballa se ci sono scenari uguali perché continua a prenderli!
        # demand_reduced, probs_reduced = FFreducer.reduce(n_scenarios=2)
        # demand_reduced[:,0]

        scenario_tree = ScenarioTree(
            name="scenario_tree",
            depth=self.time_horizon,
            branching_factors=self.branching_factors,
            dim_observations=self.env.n_items,
            initial_value=self.stoch_model.generate_scenario(),
            stoch_model=FFreducer
        )
        '''
        FFreducer = Fast_forward_W2(
            selected_data
        )
        demand_reduced, probs_reduced = FFreducer.reduce(n_scenarios=4)
        demand_reduced[:,0]

        scenario_tree = ScenarioTree(
            name='tree1',
            branching_factors=self.branching_factors,
            dim_observations=instance.n_items,
            initial_value=np.array([1]*instance.n_items),
            stoch_model=FFreducer
        )
        '''
        self.prb = MultistageOptimization(env, scenario_tree)

    def get_action(self, obs, debug=False):
        # 1. update data
        # new_scenario_tree = ScenarioTree(
        #     name="scenario_tree",
        #     depth=self.time_horizon,
        #     branching_factors=self.branching_factors,
        #     dim_observations=self.env.n_items,
        #     initial_value=obs['demand'],
        #     stoch_model=self.stoch_model
        # )
        self.prb.update_data(obs)
        # 2. solve the problem
        _, sol, _ = self.prb.solve(debug_model=debug)
        return sol

    def save_model(self, file_path):
        pass
    
    def load_model(self, file_path):
        pass

    def plot_policy(self, file_name=None):
        self._check_requisite()	
        possible_states = self.env.n_items + 1
        cmap = plt.cm.get_cmap('viridis', 3) 
        self.policy = np.zeros(
            shape=(
                possible_states,
                self.env.max_inventory_level[0] + 1,
                self.env.max_inventory_level[1] + 1
            )
        )
        for setup in range(possible_states):
            for inv0 in range(self.env.max_inventory_level[0] + 1):
                for inv1 in range(self.env.max_inventory_level[1] + 1):
                    obs = {
                        'demand': np.array([0, 0]), # does not influence
                        'inventory_level': [inv0, inv1],
                        'machine_setup': [setup]
                    }
                    self.policy[setup, inv0, inv1] = self.get_action(
                        obs
                    )

        fig, axs = plt.subplots(1, possible_states)
        fig.suptitle('Found Policy')	
        for i, ax in enumerate(axs):	
            ax.set_title(f'Setup {i}')
            im = ax.pcolormesh(	
                self.policy[i,:,:], cmap = cmap,edgecolors='k', linewidth=2	
            )	
            im.set_clim(0, possible_states - 1)	
            if i == 0:	
                ax.set_ylabel('I1')	
            ax.set_xlabel('I2')	
        # COLOR BAR:	
        # cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])	
        # fig.colorbar(im, cax=cbar_ax)	
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
        plt.close()
	
    def _check_requisite(self):	
        if self.env.n_machines > 1:	
            raise Exception('ValueIteration is defined for one machine environment')
