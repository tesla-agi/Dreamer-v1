import torch
import numpy as np
from tqdm import tqdm
import os
from replay_buffer import ReplayBuffer
from world_model import WorldModel

def train_world_model(
        buffer_path='data/random_buffer.npz',
        checkpoint_path='checkpoint/world_model.pth',
        num_iterations=20000,
        batch_size=50,
        seq_len=50,
        lr=3e-4,
        kl_weight=0.1,
        log_every=100,
        save_every=1000,
        grad_clip=100.0
):
    device="mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device:{device}")

    os.makedirs(os.path.dirname(checkpoint_path),exist_ok=True)

    print("Loading replay buffer...")
    buffer=ReplayBuffer(obs_dim=3,action_dim=1,max_episodes=100,max_steps=200)
    buffer.load(buffer_path)
    print(f"Loaded  {len(buffer)} episodes.")

    print("Creating World Model...")
    wm=WorldModel(
        obs_dim=3,
        action_dim=1,
        hidden_dim=256,
        s_dim=32,
        embed_dim=200,
        min_std=0.1,
    ).to(device)

    num_params=sum(p.numel() for p in wm.parameters())
    print(f"Total number of parameters: {num_params}")

    optimizer=torch.optim.Adam(wm.parameters(),lr=lr)

    print("Training World Model...")
    for t in tqdm(range(num_iterations)):
        obs_np,action_np,reward_np=buffer.sample_sequences(batch_size,seq_len)
        obs=torch.from_numpy(obs_np).to(device)
        action=torch.from_numpy(action_np).to(device)
        reward=torch.from_numpy(reward_np).to(device)

        optimizer.zero_grad()
        losses=wm.compute_loss(obs,action,reward,kl_weight=kl_weight)
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(wm.parameters(),grad_clip)
        optimizer.step()
        if t%log_every==0:
            print(f"Iter {t:5d} | "
                  f"total: {losses['total_loss'].item():.4f} | "
                  f"recon: {losses['recon_loss'].item():.4f} | "
                  f"reward: {losses['reward_loss'].item():.4f} | "
                  f"kl: {losses['kl_loss'].item():.4f}")

        if t%save_every==0:
            torch.save(wm.state_dict(),checkpoint_path)


    torch.save(wm.state_dict(),checkpoint_path)
    print(f"\nTraining complete! Final model saved to {checkpoint_path}")


if __name__ == "__main__":
    train_world_model()