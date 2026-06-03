import torch
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self,latent_dim=288,hidden_dim=256):
        super(Critic,self).__init__()
        self.latent_dim=latent_dim
        self.hidden_dim=hidden_dim

        self.fc1=nn.Linear(latent_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,1)

    def forward(self,h,s):
        latent=torch.cat([h,s],dim=-1)
        x=F.relu(self.fc1(latent))
        x=F.relu(self.fc2(x))
        x=self.fc3(x).squeeze(-1)
        return x


if __name__ == "__main__":
    print("=" * 60)
    print("CRITIC TEST")
    print("=" * 60)

    critic = Critic()

    B = 8
    h = torch.randn(B, 256)
    s = torch.randn(B, 32)

    v = critic(h, s)

    print(f"\n✓ Value shape: {v.shape}")
    print(f"Value:{v}")
    print(f"  values: {v[:3].tolist()}")

    n_params = sum(p.numel() for p in critic.parameters())
    print(f"\n✓ Total params: {n_params:,}")

    print("\n" + "=" * 60)
    print("CRITIC WORKS ✅")
    print("=" * 60)