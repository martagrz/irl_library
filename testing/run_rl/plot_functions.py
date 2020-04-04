import numpy as np
import matplotlib.pyplot as plt


def plot_policies(algo_stats, algo_name, col1, col2):
    algo_mean, algo_max, algo_min = algo_stats
    plt.plot(time_range,reward,label='Optimal')
    plt.plot(time_range, algo_mean, label=f"{algo_name} median",color=col1)
    plt.fill_between(time_range, algo_max, algo_min, color=col2, label=f"Confidence interval")
    plt.title(f"{algo_name} rewards over finite horizon")
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.legend(loc = 'lower right')
    plt.savefig(f'figures/{algo_name}_reward.png')
    plt.close()