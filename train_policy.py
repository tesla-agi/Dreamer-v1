import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import os
from replay_buffer import ReplayBuffer
from world_model import WorldModel
from actor import Actor
from critic import Critic
from imagine import imagine_rollout,lambda_returns

def train_policy(
        buffer_path="data/random_buffer.npz",
        world_model_path="checkpoint/world_model.pth",
        actor_path="checkpoint/actor.pth",
        critic_path="checkpoint/critic.pth",
        num_iterations=10000,
        batch_size=50,
        imagine_horizon=15,
        actor_lr=8e-5,
        critic_lr=3e-4,
        gamma=0.99,
        lam=0.95,
        entropy_weight=1e-3,
        grad_clip=100.0,
        log_every=100,
        save_every=1000
):

    device="mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(os.path.dirname(actor_path), exist_ok=True)
    os.makedirs(os.path.dirname(critic_path), exist_ok=True)

    buffer=ReplayBuffer(obs_dim=3,action_dim=1,max_episodes=100,max_steps=200)
    buffer.load(buffer_path)
    print(f"Loaded {len(buffer)} episodes")

    wm=WorldModel(obs_dim=3,action_dim=1,hidden_enc=200,hidden_dim=256,s_dim=32,embed_dim=200,min_std=0.1).to(device)
    wm.load_state_dict(torch.load(world_model_path,map_location=device))
    wm.eval()
    for param in wm.parameters():
        param.requires_grad_(False)
    print("World Model loaded and frozen")

    print("Creating actor and critic")
    actor=Actor(
        latent_dim=288,
        hidden_dim=256,
        action_dim=1,
        action_bound=2.0,
        min_std=0.1,
    ).to(device)

    critic=Critic(
        latent_dim=288,
        hidden_dim=256,
    ).to(device)

    actor_params=sum(p.numel() for p in actor.parameters())
    critic_params=sum(p.numel() for p in critic.parameters())
    print(f"Actor params: {actor_params}")
    print(f"Critic params: {critic_params}")

    actor_optimizer=torch.optim.Adam(actor.parameters(),lr=actor_lr)
    critic_optimizer=torch.optim.Adam(critic.parameters(),lr=critic_lr)

    for t in tqdm(range(num_iterations)):
        obs_np,action_np,_=buffer.sample_sequences(batch_size,seq_len=50)
        obs=torch.from_numpy(obs_np).to(device)
        action=torch.from_numpy(action_np).to(device)

        with torch.no_grad():
            wm_out=wm.observe(obs,action)
            h_real=wm_out["h_seq"]
            s_real=wm_out["s_seq"]

        B,T=h_real.shape[0],h_real.shape[1]
        start_h=h_real.reshape(B*T,-1)
        start_s=s_real.reshape(B*T,-1)
        out=imagine_rollout(
            world_model=wm,
            actor=actor,
            critic=critic,
            start_h=start_h,
            start_s=start_s,
            horizon=imagine_horizon,
        )

        with torch.no_grad():
            final_h=out["h_seq"][-1]
            final_s=out["s_seq"][-1]
            bootstrap_value=critic(final_h,final_s)

        extended_values=torch.cat([out["value_seq"],bootstrap_value.unsqueeze(0)],dim=0)
        returns=lambda_returns(
            rewards=out["reward_seq"],
            values=extended_values,
            gamma=gamma,
            lam=lam,
        )
        critic_loss=F.mse_loss(out["value_seq"],returns.detach())
        advantage=(returns-out["value_seq"]).detach()
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        actor_loss=-(out["log_prob_seq"]*advantage).mean()

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        total_loss=critic_loss+actor_loss
        total_loss.backward()
        #ctor_loss.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm_(actor.parameters(),grad_clip)
        torch.nn.utils.clip_grad_norm_(critic.parameters(),grad_clip)
        actor_optimizer.step()
        critic_optimizer.step()

        if t % log_every==0:
            print(f"Iter {t:5d} | "
                  f"actor: {actor_loss.item():.4f} | "
                  f"critic: {critic_loss.item():.4f} | "
                  f"return: {returns.mean().item():.4f}")

        if t % save_every == 0:
            torch.save(actor.state_dict(),actor_path)
            torch.save(critic.state_dict(),critic_path)

    torch.save(actor.state_dict(),actor_path)
    torch.save(critic.state_dict(),critic_path)
    print("Completed Policy Training")



if __name__=="__main__":
    train_policy()

