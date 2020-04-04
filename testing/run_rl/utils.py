import numpy as np
from algos.q_learning import QLearning
from algos.sarsa import SARSA
from algos.mc import MonteCarlo


def get_dist(a):
    a_mean = np.median(a,axis=0)
    a_max = np.quantile(a,0.75,axis=0)
    a_min = np.quantile(a,0.25,axis=0)
    return a_mean, a_max, a_min


def train(episodes,env,alpha,gamma,epsilon):

    q_algo = QLearning(env,alpha,gamma,epsilon)
    q_table, q_cum_rewards = q_algo.get_q_table(episodes)
    q_rewards, q_stock, q_policy = q_algo.evaluate_agent(1)

    sarsa = SARSA(env,alpha,gamma,epsilon)
    sarsa_q_table, sarsa_cum_rewards = sarsa.get_q_table(episodes)
    sarsa_rewards = sarsa.evaluate_agent(1)

    mc = MonteCarlo(env,alpha,gamma,epsilon)
    mc_q_table, mc_cum_rewards = mc.get_q_table(episodes)
    mc_rewards = mc.evaluate_agent(1)

    return q_cum_rewards, q_rewards, sarsa_cum_rewards, sarsa_rewards, mc_cum_rewards, mc_rewards


def run(n_runs, episodes, env, alpha, gamma, epsilon):
    q_rew = np.zeros((n_runs,env.timesteps-1))
    sarsa_rew = np.zeros((n_runs,env.timesteps-1))
    mc_rew = np.zeros((n_runs,env.timesteps-1))

    q_convergence = np.zeros((n_runs,episodes))
    sarsa_convergence = np.zeros((n_runs,episodes))
    mc_convergence = np.zeros((n_runs,episodes))


    for _ in range(n_runs):
        print('Run:', _,"/",n_runs-1)
        q_cum_rewards, q_rewards, sarsa_cum_rewards, sarsa_rewards, mc_cum_rewards, mc_rewards = train(episodes,env,alpha,gamma,epsilon)
        q_rew[_,:] = q_rewards
        sarsa_rew[_,:] = sarsa_rewards
        mc_rew[_,:] = mc_rewards

        q_convergence[_,:] = q_cum_rewards
        sarsa_convergence[_,:] = sarsa_cum_rewards
        mc_convergence[_,:] = mc_cum_rewards


    return q_rew, q_rewards, q_convergence, sarsa_rew, sarsa_rewards, sarsa_convergence, mc_rew, mc_rewards, mc_convergence
