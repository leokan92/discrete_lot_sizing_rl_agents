# -*- coding: utf-8 -*-
from models import *


class PerfectInfoAgent():
    def __init__(self, env, settings):
        super(PerfectInfoAgent, self).__init__()
        self.env = env
        self.solver = PerfectInfoOptimization(env)
        _, self.sol, _ = self.solver.solve()
        self.sol = self.sol.astype(int)

    def learn(self, epochs = 1000):
        pass

    def get_action(self, obs):
        return list(self.sol[:,self.env.current_step])
