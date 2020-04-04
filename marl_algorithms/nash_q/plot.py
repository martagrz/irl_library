import numpy as np
import matplotlib.pyplot as plt

def quiver_plot(env, policy, name):
    string = './figures/optimal_policy_{}.png'.format(name)
    string = str(string)
    fig, ax = plt.subplots(1, env.n_players)
    for player in range(env.n_players):
        grid_size = len(policy[player])
        X = np.arange(0, env.n_rows, 1)
        Y = np.arange(0, env.n_cols, 1)
        Y = np.flip(Y)

        # optimal policy maps onto U,V pairs
        # left = 0 => U,V = (-1,0)
        # right = 1 => U,V = (1,0)
        # up = 2 => U,V = (0,1)
        # down = 3 => U,V = (0,-1)
        # stay = 4 => U,V = (0,0)

        init_state_x = env.init_states[player][1]
        init_state_y = env.n_rows - 1 - env.init_states[player][0]
        goal_state_x = env.goal_states[player][1]
        goal_state_y = env.n_rows - 1 - env.goal_states[player][0]

        U, V = np.arange(0, grid_size, 1), np.arange(0, grid_size, 1)
        for state in range(grid_size):
            action = policy[player][state]
            if action == 0:
                U[state] = -1
                V[state] = 0
            if action == 1:
                U[state] = 1
                V[state] = 0
            if action == 2:
                U[state] = 0
                V[state] = 1
            if action == 3:
                U[state] = 0
                V[state] = -1
            if action == 4:
                U[state] = 0
                V[state] = 0

        U = U.reshape((env.n_rows, env.n_cols))
        V = V.reshape((env.n_rows, env.n_cols))
        ap = ax[player]
        ap.set_title('Player {}'.format(player))
        ap.quiver(X, Y, U, V, scale=25)
        ap.scatter(init_state_x, init_state_y, marker="s", label = 'Start')
        ap.scatter(goal_state_x, goal_state_y, marker="^", label = 'Goal')
        ap.legend()
    plt.savefig(string, dpi=500)
    plt.close()
