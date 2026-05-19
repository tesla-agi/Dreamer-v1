import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self,latent_dim=288,hidden_dim=200,obs_dim=3):
        super(Decoder,self).__init__()
        self.fc1=nn.Linear(latent_dim,hidden_dim)
        self.fc2=nn.Linear(hidden_dim,hidden_dim)
        self.fc3=nn.Linear(hidden_dim,obs_dim)

    def forward(self,z):
        h=F.relu(self.fc1(z))
        h=F.relu(self.fc2(h))
        obs_recon=self.fc3(h)
        return obs_recon


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("DECODER TEST")
    print("=" * 60)

    # Create decoder
    decoder = Decoder(latent_dim=288, hidden_dim=200, obs_dim=3)
    print(f"\n✓ Created decoder")

    # Count parameters
    num_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Total parameters: {num_params:,}")

    # Test 1: Single latent
    latent = torch.randn(288)
    print(f"\n✓ Single latent:")
    print(f"  Input shape:  {latent.shape}")

    obs_recon = decoder(latent)
    print(f"  Output shape: {obs_recon.shape}")

    # Test 2: Batch of latents
    latent_batch = torch.randn(32, 288)
    print(f"\n✓ Batch of latents:")
    print(f"  Input shape:  {latent_batch.shape}")

    obs_batch = decoder(latent_batch)
    print(f"  Output shape: {obs_batch.shape}")

    # Test 3: Sequence of latents
    latent_seq = torch.randn(32, 50, 288)
    print(f"\n✓ Batch of sequences:")
    print(f"  Input shape:  {latent_seq.shape}")

    obs_seq = decoder(latent_seq)
    print(f"  Output shape: {obs_seq.shape}")

    print(f"\n✅ Decoder works for all shapes!")
    print("=" * 60)