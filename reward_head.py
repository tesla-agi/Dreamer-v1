import torch
import torch.nn.functional as F
import torch.nn as nn

class RewardHead(nn.Module):
    def __init__(self,latent_dim=288,hidden_dim=200):
        super(RewardHead, self).__init__()

        self.fc1=nn.Linear(latent_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,1)

    def forward(self,z):
        h=F.relu(self.fc1(z))
        h=F.relu(self.fc2(h))
        reward=self.fc3(h).squeeze(-1)
        return reward

if __name__ == "__main__":
    head = RewardHead()
    latent = torch.randn(32, 50, 288)
    reward = head(latent)
    print(f"Input:  {latent.shape}")
    print(f"Output: {reward.shape}")  # (32, 50)
    print(f"Params: {sum(p.numel() for p in head.parameters()):,}")