# process reward_32_0.0001_0.99_0.95.npy

import numpy as np
import matplotlib.pyplot as plt

def plot_rewards(batch_size, learning_rate, gamma, gae_lambda):
    reward_name = "tuning/rewards/ppo" + str(batch_size) + "_" + str(learning_rate) + "_" + str(gamma) + "_" + str(gae_lambda) + ".npy"
    rewards = np.load(reward_name)
    timesteps_name = "tuning/timesteps/ppo" + str(batch_size) + "_" + str(learning_rate) + "_" + str(gamma) + "_" + str(gae_lambda) + ".npy"
    timesteps = np.load(timesteps_name)

    plt.figure()
    plt.plot(rewards)

    # Add title and labels
    plt.title('Rewards over episodes')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    # Save the plot as a PNG file
    save_name = "tuning/reward_plot/ppo" + str(batch_size) + "_" + str(learning_rate) + "_" + str(gamma) + "_" + str(gae_lambda) + ".png"
    plt.savefig(save_name)

    # reset the plot
    plt.figure()

    plt.plot(timesteps)
    plt.title('Timesteps Lasted over episodes')
    plt.xlabel('Episode')
    plt.ylabel('Timesteps')
    save_name = "tuning/timesteps_plot/ppo" + str(batch_size) + "_" + str(learning_rate) + "_" + str(gamma) + "_" + str(gae_lambda) + ".png"
    plt.savefig(save_name)
