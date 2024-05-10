# process reward_32_0.0001_0.99_0.95.npy

import numpy as np
import matplotlib.pyplot as plt

for batch_size in [32]:
    for learning_rate in [0.0001]:
        for gamma in [0.99, 0.95, 0.9]:
            for gae_lambda in [0.95, 0.9, 0.8]:
                reward_name = "reward_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(gamma) + "_" + str(gae_lambda) + ".npy"
                rewards = np.load(reward_name)
                #rewards = np.load("reward_32_0.0001_0.99_0.95.npy")

                plt.plot(rewards)

                # Add title and labels
                plt.title('Rewards over time')
                plt.xlabel('Time step')
                plt.ylabel('Reward')

                # Save the plot as a PNG file
                save_name = "rewards_plot_" + str(batch_size) + "_" + str(learning_rate) + "_" + str(gamma) + "_" + str(gae_lambda) + ".png"
                plt.savefig(save_name)

                # Display the plot
                plt.show()