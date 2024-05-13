import numpy as np
import matplotlib.pyplot as plt

# Load the data from the .npy files
rewards = np.load('tuning/rewards/pybullet_ppo.npy')
timesteps = np.load('tuning/timesteps/pybullet_ppo.npy')

def sliding_average(data, window_size=100):
    """
    Calculate the sliding average of the given data.
    
    Parameters:
    data (array-like): The data to calculate the sliding average for.
    window_size (int): The size of the sliding window.
    
    Returns:
    array-like: The sliding average of the data.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Calculate the sliding averages
sliding_avg_rewards = sliding_average(rewards, window_size=100)
sliding_total_rewards = sliding_average(rewards * timesteps, window_size=100)
sliding_avg_timesteps = sliding_average(timesteps, window_size=100)

# Generate the plots
fig, axs = plt.subplots(3, 1, figsize=(10, 10))

# # No sliding average rewards per 100 episodes
# axs[0].plot(rewards, label='Average Reward per step for each episode')
# axs[0].set_title('Average Reward per step for each episode')
# axs[0].set_xlabel('Episode')
# axs[0].set_ylabel('Total Reward')
# axs[0].legend()

# Plot the sliding average total reward per 100 episodes
axs[0].plot(sliding_avg_rewards, label='Sliding Average of Average Reward per episode (100 episodes)')
axs[0].set_title('Sliding Average of Average Reward per episode (Survival Reward = 0, PPO)')
axs[0].set_xlabel('Episode')
axs[0].set_ylabel('Average Episode Reward')
axs[0].legend()

# Plot the sliding average total reward per 100 episodes
axs[1].plot(sliding_total_rewards, label='Sliding Average Total Reward (100 episodes)', color='orange')
axs[1].set_title('Sliding Average of Total Reward per episode (Survival Reward = 0, PPO)')
axs[1].set_xlabel('Episode')
axs[1].set_ylabel('Total Episode Reward')
axs[1].legend()

# Plot the sliding average total timesteps per 100 episodes
axs[2].plot(sliding_avg_timesteps, label='Sliding Average Total Timesteps (100 episodes)', color='green')
axs[2].set_title('Sliding Average of Total Reward per episode (Survival Reward = 0, PPo)')
axs[2].set_xlabel('Episode')
axs[2].set_ylabel('Total Episode Timesteps')
axs[2].legend()

plt.tight_layout()
plt.show()