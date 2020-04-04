import numpy as np
from env import NGridworld
from nashq import NashQLearning
from plot import quiver_plot
import matplotlib.pyplot as plt


def run(env, episodes, alpha, gamma, epsilon, error_variance, name, error=False):
    algo = NashQLearning(env, alpha, gamma, epsilon, error_variance, error)
    q_table, optimal_policy, number_steps = algo.train(episodes)
    quiver_plot(env, optimal_policy, name)
    return number_steps


# Set parameters
np.random.seed(0)
n_players = 2
n_rows = 4
n_cols = 4
init_states = np.array(((0, 0), (0, 2)))
goal_states = np.array(((3, 1), (3, 3)))
alpha = 0.2
gamma = 0.6
epsilon = 0.2
episodes = 1000
plot_steps = 10
error_variance = [0, 1]

env = NGridworld(n_players, n_rows, n_cols, init_states, goal_states)

n_steps = np.zeros((len(error_variance), episodes))
for i in range(len(error_variance)):
    print('\n', 'Run', i, '/', len(error_variance)-1)
    variance = error_variance[i]
    n_steps[i, :] = run(env, episodes, alpha, gamma, epsilon, variance, i, error=True)

for i in range(len(error_variance)):
    plt.plot(range(episodes)[::plot_steps], n_steps[i, :][::plot_steps], label='Error = {}'.format(error_variance[i]))

plt.xlabel('Episode')
plt.ylabel('Number of steps')
plt.legend()
plt.savefig('./figures/n_steps_convergence.png', dpi=300)
plt.close()
