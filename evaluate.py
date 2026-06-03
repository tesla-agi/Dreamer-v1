import torch
import numpy as np
import gymnasium as gym
from world_model import WorldModel
from actor import Actor

def evaluate(
        world_model_path="checkpoint/world_model.pth",
        actor_path="checkpoint/actor.pth",
        num_episodes=10,
        max_steps=200,
        render=False,
        seed=42

):
    device="mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(seed)
    np.random.seed(seed)

    wm=WorldModel(
        obs_dim=3,
        action_dim=1,
        hidden_enc=200,
        hidden_dim=256,
        s_dim=32,
        embed_dim=200,
        min_std=0.1,
    ).to(device)
    wm.load_state_dict(torch.load(world_model_path,map_location=device))
    wm.eval()

    actor=Actor(latent_dim=288,hidden_dim=256,action_dim=1,action_bound=2,min_std=0.1).to(device)
    actor.load_state_dict(torch.load(actor_path,map_location=device))
    actor.eval()

    env=gym.make("Pendulum-v1",render_mode="human" if render else None)
    episode_reward=[]

    for ep in range(num_episodes):
        obs,_=env.reset(seed=seed+ep)
        h,s=wm.rssm.init_state(batch_size=1,device=device)
        a_zeros=torch.zeros(1,1,device=device)
        total_reward=0

        for t in range(max_steps):
            obs_t=torch.from_numpy(obs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                obs_embed=wm.encoder(obs_t)
                h,s,_,_=wm.rssm.obs_step(h_prev=h,a_prev=a_zeros,s_prev=s,obs_embed=obs_embed)
                action_dist=actor(h,s)
                action=action_dist.mean

            action_np=action.cpu().numpy().flatten().astype(np.float32)

            obs,reward,terminated,truncated,_=env.step(action_np)
            total_reward+=reward
            a_zeros=action

            if terminated or truncated:
                break

        episode_reward.append(total_reward)
        print(f"Episode {ep + 1}: reward = {total_reward:.2f}")

    env.close()

    mean_reward = np.mean(episode_reward)
    std_reward = np.std(episode_reward)

    print(f"\n{'=' * 60}")
    print(f"EVALUATION RESULTS ({num_episodes} episodes)")
    print(f"{'=' * 60}")
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Random baseline: -1239")
    print(f"{'=' * 60}")

    return mean_reward


if __name__ == "__main__":
    evaluate()





