import numpy as np


class SendKeep:
    def __init__(self):
        self.actions = np.array((0,1)) #0: keep, 1:send
        self.states = np.array((0,1))
        self.n_states = len(self.states)
        self.n_actions = len(self.actions)
        self.reset = np.random.choice((0, 1))
        self.n_players = 2

    def step(self, state, action):
        done = False
        rewards = np.zeros(self.n_players)
        if action == 0:
            if state == 0:
                next_state = 0
                rewards[0] = 1
                rewards[1] = 0
            if state == 1:
                next_state = 1
                rewards[0] = 3
                rewards[1] = 2
        else:
            if state == 0:
                next_state = 1
                rewards[0] = 0
                rewards[1] = 3
            else:
                next_state = 0
                rewards[0] = 0
                rewards[1] = 0

        return next_state, rewards, done
