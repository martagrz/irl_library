import numpy as np

class NGridworld():
    def __init__(self, n_players, n_rows, n_cols, init_states, goal_states):
        self.n_players = n_players
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.grid_size = self.n_rows * self.n_cols
        self.init_states = init_states
        self.goal_states = goal_states
        self.actions = np.arange(5) #0:left, 1:right, 2:up, 3:down, 4: stay

    def get_next_state(self, player_state, player_action):
        player_row = player_state[0]
        player_col = player_state[1]
        if player_action == 0:
            if player_col == 0:
                pass
            else:
                player_col -= 1

        elif player_action == 1:
            if player_col == self.n_cols-1:
                pass
            else:
                player_col += 1

        elif player_action == 2:
            if player_row == 0:
                pass
            else:
                player_row -= 1

        elif player_action == 3:
            if player_row == self.n_rows-1:
                pass
            else:
                player_row += 1

        elif player_action == 4:
            pass

        player_row = np.int(player_row)
        player_col = np.int(player_col)
        return np.array((player_row, player_col))


    def step(self, current_state, actions):
        next_state = np.zeros((self.n_players, 2))
        rewards = np.ones(self.n_players)
        done = np.zeros(self.n_players)

        for n in range(self.n_players):
            player_state = current_state[n]
            player_action = actions[n]
            player_next_state = self.get_next_state(player_state, player_action)

            if np.all(player_next_state == self.goal_states[n]):
                rewards[n] = 10
                done[n] = 1
            else:
                rewards[n] = 0

            if actions[n] == 4:
                rewards[n] = 0

            next_state[n, :] = player_next_state

        for n in range(self.n_players):
            for m in range(self.n_players):
                if n != m:
                    if np.all(next_state[n] == next_state[m]): #This works for two players ONLY
                        next_state[n] = current_state[n]
                        next_state[m] = current_state[m]

        return next_state, rewards, done

    def reset(self):
        return self.init_states