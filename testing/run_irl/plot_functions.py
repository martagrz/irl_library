import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_quiver(n_rows, n_cols, policy, title=None, show=False):
    n_states = n_rows * n_cols
    X = np.arange(0, n_cols, 1)
    Y = np.arange(0, n_rows, 1)
    Y = np.flip(Y)

    # left = 0 => U,V = (-1,0)
    # right = 1 => U,V = (1,0)
    # up = 2 => U,V = (0,1)
    # down = 3 => U,V = (0,-1)
    # stay = 4 => U,V = (0,0)
    U, V = np.arange(0, n_states, 1), np.arange(0, n_states, 1)
    for state in np.arange(len(policy)):
        action = policy[state]
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

    U = U.reshape((n_rows, n_cols))
    V = V.reshape((n_rows, n_cols))
    fig, ax = plt.subplots()
    q = ax.quiver(X, Y, U, V, scale=25)
    if title is not None:
        plt.savefig(title)
    if show:
        plt.show()
    plt.close()


def plot_reward_surface(n_rows, n_cols, reward_vector, title=None, show=False):
    x = np.arange(0, n_cols, 1)
    y = np.arange(0, n_rows, 1)
    y = np.flip(y)
    X, Y = np.meshgrid(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Z = reward_vector
    Z = Z.reshape((n_rows, n_cols))
    ax.plot_surface(X, Y, Z)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    if title is not None:
        plt.savefig(title)
    if show:
        plt.show()
    plt.close()
