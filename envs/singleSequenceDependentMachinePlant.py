# -*- coding: utf-8 -*-
import gym
import numpy as np
from envs.simplePlant import SimplePlant


class SequenceDependentPlant(SimplePlant):
    def __init__(self, settings, stoch_model):
        super(SequenceDependentPlant, self).__init__(settings, stoch_model)
        

    def _next_observation(self):
        """
        Returns the next observation after all the setup updates and demand subtraction
        """
        self.demand = self.stoch_model.generate_scenario()
        return self.demand
    
    def _take_action(self, action):
        """
        This method needs to return the cost on each lot decision devided in three main costs:
        
        Inputs
        ----------
            -action: action taken by the agent
    
        Returns
        -------
            - state updated component: the new inventory, machine setup, and effective setup
                - next inventory level: the inventory level changes with the demand, lost-setup production, production
                - next machine setup: gives the next machine set (usefull when we have setup time)
                - next effective setup: the setup that will be used for the production (usefull when we have setup time)
            - total_cost: the sum of all costs
            - next setup time counter: used to control the setup time
          
        """
        self.total_cost = np.array([0,0,0,0,0])

        setup_costs = np.zeros(self.n_machines)
        setup_loss = np.zeros(self.n_machines)
        production_costs = np.zeros(self.n_machines)
        lost_sales = np.zeros(self.n_items)
        holding_costs = np.zeros(self.n_items)

        # if we are just changing the setup, we use the setup cost matrix with the corresponding position given by the actual setup and the new setup
        for m in range(0, self.n_machines):            
            # 1. IF NEEDED CHANGE SETUP
            setup_costs[m] = self.setup_costs[m][self.machine_setup[m]][action[m]]
            setup_loss[m] = self.setup_loss[m][self.machine_setup[m]][action[m]]
            self.machine_setup[m] = action[m]
            if action[m] != 0: # if the machine is not iddle
                # 2. PRODUCTION
                self.inventory_level[action[m] - 1] += self.machine_production_matrix[m][action[m] - 1] - setup_loss[m]
                production_costs[m] = self.production_costs[m][action[m] - 1]
        
        # 3. SATIFING DEMAND
        for i in range(0, self.n_items):
            self.inventory_level[i] -= self.demand[i]
            if self.inventory_level[i] < 0:
                lost_sales[i] = - self.inventory_level[i] * self.lost_sales_costs[i]
                self.inventory_level[i] = 0
            # 4. HOLDING COSTS
            holding_costs[i] += self.inventory_level[i] * self.holding_costs[i]

        self.total_cost = {
            'setup_costs': sum(setup_costs),
            'production_costs': sum(production_costs),
            'lost_sales': sum(lost_sales),
            'holding_costs': sum(holding_costs),
        }
