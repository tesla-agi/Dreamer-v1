import numpy as np
from replay_buffer import ReplayBuffer

"""Quick test of the replay buffer."""

print("=" * 60)
print("REPLAY BUFFER TEST")
print("=" * 60)

# Create buffer
buffer = ReplayBuffer(obs_dim=3, action_dim=1, max_episodes=10)
print(f"\n✓ Created buffer")
print(f"  Length: {len(buffer)}")

# Make a fake episode (just random data for testing)
fake_obs = [np.random.randn(3) for _ in range(201)]
fake_actions = [np.random.randn(1) for _ in range(200)]
fake_rewards = [np.random.randn() for _ in range(200)]

# Add it
buffer.add_episode(fake_obs, fake_actions, fake_rewards)
print(f"\n✓ Added 1 episode")
print(f"  Length: {len(buffer)}")

# Add a few more
for _ in range(4):
    fake_obs = [np.random.randn(3) for _ in range(201)]
    fake_actions = [np.random.randn(1) for _ in range(200)]
    fake_rewards = [np.random.randn() for _ in range(200)]
    buffer.add_episode(fake_obs, fake_actions, fake_rewards)

print(f"\n✓ Added 4 more episodes")
print(f"  Length: {len(buffer)}")

# Sample some sequences
obs_batch, action_batch, reward_batch = buffer.sample_sequences(
    batch_size=8,
    seq_len=20
)

print(f"\n✓ Sampled sequences:")
print(f"  obs_batch shape:    {obs_batch.shape}")
print(f"  action_batch shape: {action_batch.shape}")
print(f"  reward_batch shape: {reward_batch.shape}")

# Verify shapes
assert obs_batch.shape == (8, 21, 3), f"Wrong obs shape!"
assert action_batch.shape == (8, 20, 1), f"Wrong action shape!"
assert reward_batch.shape == (8, 20), f"Wrong reward shape!"
print(f"\n✓ All shapes correct!")
print("\n" + "=" * 60)
print("BUFFER WORKS! ✅")
print("=" * 60)
