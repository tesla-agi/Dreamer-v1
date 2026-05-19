import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

env=gym.make('Pendulum-v1')

obs,info=env.reset(seed=42)

all_actions=[]
all_obs=[obs]
all_rewards=[]

done=False
step=0

while not done:
    action=env.action_space.sample()
    obs,reward,terminated,truncated,info=env.step(action)
    all_actions.append(action)
    all_obs.append(obs)
    all_rewards.append(reward)

    done=terminated or truncated
    step+=1

env.close()

all_obs=np.array(all_obs)
all_rewards=np.array(all_rewards)
all_actions=np.array(all_actions)

cos_theta=all_obs[:,0]
sin_theta=all_obs[:,1]
omega=all_obs[:,2]

theta_radian=np.arctan2(sin_theta,cos_theta)
theta_degree=np.degrees(theta_radian)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plot 1: Angle over time
axes[0, 0].plot(theta_degree)
axes[0, 0].set_title('Pendulum Angle (degrees)')
axes[0, 0].set_xlabel('Step')
axes[0, 0].set_ylabel('Angle from vertical')
axes[0, 0].axhline(y=0, color='g', linestyle='--', label='Target (0°)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: Angular velocity over time
axes[0, 1].plot(omega, color='orange')
axes[0, 1].set_title('Angular Velocity')
axes[0, 1].set_xlabel('Step')
axes[0, 1].set_ylabel('ω (rad/s)')
axes[0, 1].axhline(y=0, color='g', linestyle='--')
axes[0, 1].grid(True)

# Plot 3: Reward over time
axes[1, 0].plot(all_rewards, color='red')
axes[1, 0].set_title('Reward at Each Step')
axes[1, 0].set_xlabel('Step')
axes[1, 0].set_ylabel('Reward')
axes[1, 0].axhline(y=0, color='g', linestyle='--', label='Max reward (0)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Plot 4: Actions over time
axes[1, 1].plot(all_actions, color='purple')
axes[1, 1].set_title('Actions (Torque)')
axes[1, 1].set_xlabel('Step')
axes[1, 1].set_ylabel('Torque')
axes[1, 1].axhline(y=0, color='g', linestyle='--')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('results/random_episode.png')
plt.show()

print("Plot saved to results/random_episode.png")