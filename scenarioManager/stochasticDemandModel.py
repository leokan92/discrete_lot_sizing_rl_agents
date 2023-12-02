# -*- coding: utf-8 -*-
import random
import numpy as np

def _custom_pmf(population, weights, size):    
    # TODO: improve this code:
    if len(size) == 2:
        ans = np.zeros(shape=size)
        for i in range(size[0]):
            ans[i,:] = random.choices(
                population=population,
                weights=weights,
                k=size[1]
            )
    else:
        ans = np.array(random.choices(
            population=population,
            weights=weights,
            k=size[0]
        ))
    return ans

def _custom_item_specific(distributions, size):
    ans = np.zeros(size)

    for i, distr in enumerate(distributions):
        if distr['name'] == 'normal':
            ans[i,:] = np.random.normal(
                distr['mu'],
                distr['sigma'],
                size=ans[i,:].shape
            ).astype(int)
        elif distr['name'] == 'discrete_uniform': 
            ans[i,:] = np.random.randint(
                low=distr['low'],
                high=distr['high'],
                size=ans[i,:].shape
            )
        elif distr['name'] =='binomial': 
            ans[i,:] = np.random.binomial(
                distr['n'],
                distr['p'],
                size=ans[i,:].shape
            )
        elif distr['name'] == 'probability_mass_function':
            ans[i,:] = _custom_pmf(
                population=distr['demand_distribution']['vals'],
                weights=distr['demand_distribution']['probs'],
                size=ans[i,:].shape
            )
    return ans


class StochasticDemandModel():
    def __init__(self, settings):
        self.settings = settings
        # DEFINITION ELEMENTAR DISTRIBUTION
        distr_dict = {
            'normal': lambda x: np.random.normal(
                self.settings['demand_distribution']['mu'],
                self.settings['demand_distribution']['sigma'],
                size=x
            ),
            'discrete_uniform': lambda x: np.random.randint(
                low=self.settings['demand_distribution']['low'],
                high=self.settings['demand_distribution']['high'],
                size=x
            ),
            'binomial': lambda x: np.random.binomial(
                self.settings['demand_distribution']['n'],
                self.settings['demand_distribution']['p'],
                size=x
            ),
            'probability_mass_function': lambda x: _custom_pmf(
                population=self.settings['demand_distribution']['vals'],
                weights=self.settings['demand_distribution']['probs'],
                size=x
            ),
            'item_specific_uniform': lambda x: _custom_item_specific(
                distributions=self.settings['demand_distribution']['distributions'],
                size=x
            )
        }
        if self.settings['demand_distribution']['name'] == 'probability_mass_function':
            if sum(self.settings['demand_distribution']['probs']) != 1:
                raise ValueError('sum of prob different than one')
        self.name_distribution = self.settings['demand_distribution']['name']
        self.generate = distr_dict[self.settings['demand_distribution']['name']]
        self.n_items = self.settings['n_items']

    def fit(self, data):
        pass

    def generate_scenario(self, history=None, n_time_steps=1):
        # if n_time_steps == 1:
        #     return self.generate( (self.n_items, ) )
        # else:
        #     return self.generate( (self.n_items, n_time_steps) )
        return self.generate( (self.n_items, n_time_steps) )
