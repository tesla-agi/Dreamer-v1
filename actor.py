import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class Actor(nn.Module):
    def __init__(self,latent_dim=288,hidden_dim=256,action_dim=1,action_bound=2,min_std=0.1):
        super(Actor,self).__init__()
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim
        self.action_dim=action_dim
        self.action_bound=action_bound
        self.min_std=min_std

        self.fc1=nn.Linear(latent_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,2*action_dim)

    def forward(self,h,s):
        latent=torch.cat([h,s],dim=-1)
        x=F.relu(self.fc1(latent))
        x=F.relu(self.fc2(x))
        x=self.fc3(x)
        mu,raw_std=torch.chunk(x,2,dim=-1)
        mu=self.action_bound*torch.tanh(mu)
        sigma=F.softplus(raw_std)+self.min_std
        return Normal(mu,sigma)


if __name__ == "__main__":
    print("=" * 60)
    print("ACTOR TEST")
    print("=" * 60)

    actor = Actor()

    # Fake input
    B = 8
    h = torch.randn(B, 256)
    s = torch.randn(B, 32)

    # Forward
    dist = actor(h, s)

    print(f"\n✓ Output distribution:")
    print(f"  mean shape:  {dist.mean.shape}")
    print(f"  std shape:   {dist.stddev.shape}")
    print(f"  mean range:  [{dist.mean.min().item():.3f}, {dist.mean.max().item():.3f}]")
    print(f"  std range:   [{dist.stddev.min().item():.3f}, {dist.stddev.max().item():.3f}]")

    # Sample
    action = dist.rsample()
    print(f"\n✓ Sampled action shape: {action.shape}")
    print(f"  action range: [{action.min().item():.3f}, {action.max().item():.3f}]")

    # Log prob
    log_prob = dist.log_prob(action)
    print(f"\n✓ Log prob shape: {log_prob.shape}")

    # Param count
    n_params = sum(p.numel() for p in actor.parameters())
    print(f"\n✓ Total params: {n_params:,}")

    print("\n" + "=" * 60)
    print("ACTOR WORKS ✅")
    print("=" * 60)