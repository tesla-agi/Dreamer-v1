import gymnasium as gym
import numpy as np

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

print("=" * 60)
print(f"Episode finished after {step} steps")
print(f"Total reward: {sum(all_rewards):.2f}")
print(f"Average reward per step: {np.mean(all_rewards):.2f}")
print(f"Best step reward: {max(all_rewards):.2f}")
print(f"Worst step reward: {min(all_rewards):.2f}")
print("=" * 60)

env.close()