import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self,obs_dim=3,hidden_dim=200,embed_dim=200):
        super(Encoder,self).__init__()
        self.fc1=nn.Linear(obs_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,embed_dim)

    def forward(self,obs):
        h=F.relu(self.fc1(obs))
        h=F.relu(self.fc2(h))
        embedding=self.fc3(h)
        return embedding


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("ENCODER TEST")
    print("=" * 60)

    # Create encoder
    encoder = Encoder(obs_dim=3, hidden_dim=200, embed_dim=200)
    print(f"\n✓ Created encoder")

    # Count parameters
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Total parameters: {num_params:,}")

    # Test 1: Single observation
    obs = torch.randn(3)  # Random observation
    print(f"\n✓ Single observation:")
    print(f"  Input shape:  {obs.shape}")

    embedding = encoder(obs)
    print(f"  Output shape: {embedding.shape}")

    # Test 2: Batch of observations
    obs_batch = torch.randn(32, 3)  # 32 observations
    print(f"\n✓ Batch of observations:")
    print(f"  Input shape:  {obs_batch.shape}")

    embed_batch = encoder(obs_batch)
    print(f"  Output shape: {embed_batch.shape}")

    # Test 3: Sequence of observations (batch, time, features)
    obs_seq = torch.randn(32, 50, 3)  # batch=32, time=50
    print(f"\n✓ Batch of sequences:")
    print(f"  Input shape:  {obs_seq.shape}")

    embed_seq = encoder(obs_seq)
    print(f"  Output shape: {embed_seq.shape}")

    print(f"\n✅ Encoder works for all shapes!")
    print("=" * 60)