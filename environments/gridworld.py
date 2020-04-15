import numpy as np


class GridWorld:
    def __init__(self, n_rows, n_cols, discount_rate, random_shift, init_state_index=None, goal_state_index=None):
        self.n_rows = n_rows
        self.n_cols = n_cols

        self.action_space = np.arange(0, 5, 1)  # 0: left, 1: right, 2: up, 3: down, 4: stay
        self.n_actions = len(self.action_space)

        self.n_states = n_rows * n_cols
        self.state_space = np.arange(self.n_states)

        self.discount_rate = discount_rate
        self.random_shift = random_shift

        if init_state_index is None:
            self.init_state_index = 0
        else:
            self.init_state_index = init_state_index

        if goal_state_index is None:
            self.goal_state_index = self.n_states-1
        else:
            self.goal_state_index = goal_state_index

        self.states_actions_rewards_matrix = -10 * np.ones((self.n_states, self.n_actions))
        self.states_actions_rewards_matrix[self.goal_state_index] = -1

        self.transition_probabilities = self.get_transition_probabilities()

    def get_transition_probabilities(self):
        transition_probabilities = np.zeros((self.n_states, self.n_actions, self.n_states))  # [s, a, s']
        for state_index in np.arange(self.n_states):
            possible_next_states = np.zeros(self.n_actions)
            _probabilities = np.zeros(self.n_actions)

            for action_index in np.arange(self.n_actions):
                next_state_index, _, _ = self.step(state_index, action_index, wind=False)
                possible_next_states[action_index] = next_state_index
                transition_probabilities[state_index, action_index, next_state_index] += 1- self.random_shift

            for next_state_index in possible_next_states:
                _prob = self.random_shift/self.n_actions
                transition_probabilities[state_index, :, np.int(next_state_index)] += _prob
        return transition_probabilities

    def reset(self):
        return self.init_state_index

    def get_coordinates(self, state_index):
        row = state_index // self.n_cols
        col = state_index % self.n_cols
        return np.array((row, col))

    def get_index(self, state):
        row = state[0]
        col = state[1]
        index = row * self.n_cols + col
        return np.int(index)

    def step(self, state_index, action, wind=True):
        state = self.get_coordinates(state_index)
        done = False
        next_state = state

        if action == 0:
            if state[1] != 0:
                next_state[1] = state[1] - 1

        elif action == 1:
            if state[1] != self.n_cols - 1:
                next_state[1] = state[1] + 1

        elif action == 2:
            if state[0] != 0:
                next_state[0] = state[0] - 1

        elif action == 3:
            if state[0] != self.n_rows-1:
                next_state[0] = state[0] + 1

        elif action == 4:
            next_state = state

        next_state_index = self.get_index(next_state)

        if wind:
            if np.random.uniform(0, 1) < self.random_shift:
                _, possible_next_states = self.get_transitions(state_index, action)
                next_state_index = np.int(np.random.choice(possible_next_states))

        if next_state_index == self.goal_state_index:
            done = True

        reward = self.states_actions_rewards_matrix[next_state_index, action]

        return next_state_index, reward, done

