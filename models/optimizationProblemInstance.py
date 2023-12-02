# -*- coding: utf-8 -*-
import numpy as np

class Instance():
    def __init__(self, setting, stoch_model):
        self.n_items = setting['n_items']
        self.initial_inventory = setting['initial_inventory']
        self.setup_costs = np.array(setting['setup_costs'])
        self.lost_sales_costs = np.array(setting['lost_sales_costs'])
        self.holding_costs = np.array(setting['holding_costs'])
        self.stoch_model = stoch_model
        self.setting = setting
    
    def update_data(self, state):
        self.initial_inventory = state['inventory']
