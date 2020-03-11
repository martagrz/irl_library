import numpy as np


class GridWorld:
    def __init__(self, n_rows, n_cols, discount_rate, random_shift=0):
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.state_space = np.arange(self.n_states)
        self.action_space = np.arange(0, 4, 1)  # 0: left, 1: right, 2: up, 3: down, 4: stay

        self.n_states = n_rows * n_cols
        self.n_actions = len(self.action_space)

        self.discount_rate = discount_rate
        self.random_shift = random_shift

        self.init_state = np.array((0, 0))
        self.goal_state = np.array((self.n_rows-1, self.n_cols-1))

    def get_transitions(self, state_index, action):
        state = self.get_coordinates(state_index)
        transition_probabilities = np.zeros(self.n_actions)
        possible_next_states = np.zeros(self.n_actions)
        for i in np.arange(self.n_actions):
            other_action = self.action_space[i]
            next_state, _, _ = self.step(state, other_action)
            possible_next_states[i] = next_state
            if other_action == action:
                transition_probabilities[i] = 1
        return transition_probabilities, possible_next_states

    def reset(self):
        return self.init_state

    def get_coordinates(self, state_index):
        row = state_index % self.n_cols
        col = state_index // self.n_cols
        return np.array((row, col))

    def get_index(self, state):
        row = state[0]
        col = state[1]
        index = row * self.n_cols + col
        return index

    def step(self, state, action):
        done = False
        reward = - 1
        next_state = state

        if action == 0:
            if state[1] != 0:
                next_state[1] = state[1] - 1

        elif action == 1:
            if state[1] != self.n_cols - 1:
                next_state[1] = state[1] + 1

        elif action == 2:
            if state[0] != 0:
                next_state[0] = state[0] - -1

        elif action == 3:
            if state[0] != self.n_rows-1:
                next_state[0] = state[0] + 1

        elif action == 4:
            next_state = state

        if next_state == self.goal_state:
            reward = 10
            done = True

        return next_state, reward, done

