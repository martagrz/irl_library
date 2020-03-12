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

        self.states_actions_rewards_matrix = -np.ones((self.n_states, self.n_actions))
        self.states_actions_rewards_matrix[self.goal_state_index] = 10

        self.transition_probabilities = np.zeros((self.n_states, self.n_actions, self.n_actions))
        self.possible_next_states = np.zeros((self.n_states, self.n_actions, self.n_actions))

        for state in range(self.n_states-1):
            for action in range(self.n_actions):
                probabilities, next_states = self.get_transitions(state, action)
                self.transition_probabilities[state, action] = probabilities
                self.possible_next_states[state, action] = next_states

    def get_transitions(self, state_index, action):
        transition_probabilities = np.zeros(self.n_actions)
        possible_next_states = np.zeros(self.n_actions)
        for i in np.arange(self.n_actions):
            other_action = self.action_space[i]
            next_state_index, _, _ = self.step(state_index, other_action, wind=False)
            possible_next_states[i] = np.int(next_state_index)
            if other_action == action:
                transition_probabilities[i] = 1 - self.random_shift + self.random_shift/self.n_actions
            else:
                transition_probabilities[i] = self.random_shift/self.n_actions
        return transition_probabilities, possible_next_states

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

        if wind:
            if np.random.uniform(0, 1) < self.random_shift:
                _, possible_next_states = self.get_transitions(state_index, action)
                next_state = np.random.choice(possible_next_states)

        next_state_index = self.get_index(next_state)

        if next_state_index == self.goal_state_index:
            done = True

        reward = self.states_actions_rewards_matrix[next_state_index, action]

        return next_state_index, reward, done

