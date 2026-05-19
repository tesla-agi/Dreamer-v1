import gymnasium as gym
import numpy as np
from tqdm import tqdm
from replay_buffer import ReplayBuffer

def collect_random_episodes(num_episodes=100,save_path='data/random_buffer.npz'):
    env=gym.make('Pendulum-v1')
    buffer=ReplayBuffer(
        obs_dim=3,
        action_dim=1,
        max_episodes=num_episodes,
        max_steps=200
    )
    all_ep_rewards=[]

    for episode in tqdm(range(num_episodes)):
        obs,info=env.reset()
        ep_obs=[obs]
        ep_action=[]
        ep_reward=[]

        done=False
        while not done:
            action=env.action_space.sample()
            obs,reward,terminated,truncated,info=env.step(action)
            ep_obs.append(obs)
            ep_action.append(action)
            ep_reward.append(reward)
            done=truncated or terminated

        buffer.add_episode(ep_obs,ep_action,ep_reward)
        all_ep_rewards.append(sum(ep_reward))

    env.close()

    # Print statistics
    print("\n" + "=" * 60)
    print("COLLECTION COMPLETE")
    print("=" * 60)
    print(f"Episodes collected: {len(buffer)}")
    print(f"Mean reward:{np.mean(all_ep_rewards):.2f}")
    print(f"Min reward:{np.min(all_ep_rewards):.2f}")
    print(f"Max reward:{np.max(all_ep_rewards):.2f}")
    print(f"Std reward:{np.std(all_ep_rewards):.2f}")

    # Save buffer to disk
    buffer.save(save_path)
    print("="*60)


if __name__ == "__main__":
    collect_random_episodes(num_episodes=100, save_path='data/random_buffer.npz')