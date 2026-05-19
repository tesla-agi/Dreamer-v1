import numpy as np

class ReplayBuffer:
    def __init__(self,obs_dim=3,action_dim=1,max_episodes=200,max_steps=200):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.max_episodes = max_episodes
        self.max_steps = max_steps

        self.observations=np.zeros(
            (self.max_episodes,self.max_steps+1,self.obs_dim)
        )

        self.actions=np.zeros(
            (self.max_episodes,self.max_steps,self.action_dim)
        )

        self.rewards=np.zeros(
            (self.max_episodes,self.max_steps)
        )

        self.num_episodes=0

    def add_episode(self,obs_list,action_list,reward_list):
        if self.num_episodes>=self.max_episodes:
            print("Episode Limit reached")
            return

        idx=self.num_episodes
        self.observations[idx]=np.array(obs_list,dtype=np.float32)
        self.actions[idx]=np.array(action_list,dtype=np.float32)
        self.rewards[idx]=np.array(reward_list,dtype=np.float32)
        self.num_episodes+=1

    def sample_sequences(self,batch_size=50,seq_len=50):
        episode_indices=np.random.randint(0,self.num_episodes,size=batch_size)
        max_start=self.max_steps-seq_len
        start_indices=np.random.randint(0,max_start,size=batch_size)
        obs_batch=np.zeros(
            (batch_size,seq_len+1,self.obs_dim),dtype=np.float32
        )
        action_batch=np.zeros(
            (batch_size,seq_len,self.action_dim),dtype=np.float32
        )
        reward_batch=np.zeros(
            (batch_size,seq_len),dtype=np.float32
        )
        for idx in range(batch_size):
            ep_idx=episode_indices[idx]
            start=start_indices[idx]
            end=start+seq_len

            obs_batch[idx]=self.observations[ep_idx,start:end+1]
            action_batch[idx]=self.actions[ep_idx,start:end]
            reward_batch[idx]=self.rewards[ep_idx,start:end]

        return obs_batch,action_batch,reward_batch


    def save(self,path):
        np.savez(
            path,
            observations=self.observations[:self.num_episodes],
            actions=self.actions[:self.num_episodes],
            rewards=self.rewards[:self.num_episodes],
            num_episodes=self.num_episodes
        )
        print(f"Saved {self.num_episodes} episodes to {path}")

    def load(self,path):
        data=np.load(path)
        self.num_episodes=int(data["num_episodes"])
        self.observations[:self.num_episodes]=data["observations"]
        self.actions[:self.num_episodes]=data["actions"]
        self.rewards[:self.num_episodes]=data["rewards"]
        print(f"Loaded {self.num_episodes} episodes from {path}")

    def __len__(self):
        return self.num_episodes


