# -*- coding: utf-8 -*-
import time

class DummyAgent():

    def __init__(self, env):
        super(DummyAgent, self).__init__()
        self.env = env

    def get_action(self, obs):
        return self.env.action_space.sample()

    def learn(self, epochs = 1000):
        start_time = time.time()
        for epoch in range(epochs):
            state = self.env.reset()
            done = False
            while not done: # Loop through the episode
                action = self.get_action()
                state, reward, done, info = self.env.step(action)
    
        time_duration = time.time - start_time
        print(time_duration)
        print("\nFinished Learning. \n")

    def save(self):
        pass
