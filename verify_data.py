"""Verify saved data loads correctly."""

from replay_buffer import ReplayBuffer

# Create empty buffer
buffer = ReplayBuffer(
    obs_dim=3,
    action_dim=1,
    max_episodes=100,
    max_steps=200
)

# Load from disk
buffer.load('data/random_buffer.npz')

print(f"\n✓ Loaded {len(buffer)} episodes")

# Check data shapes
print(f"\nData shapes:")
print(f"  observations: {buffer.observations.shape}")
print(f"  actions:      {buffer.actions.shape}")
print(f"  rewards:      {buffer.rewards.shape}")

# Sample some sequences
obs, act, rew = buffer.sample_sequences(batch_size=4, seq_len=10)
print(f"\nSampled batch:")
print(f"  obs shape:    {obs.shape}")
print(f"  action shape: {act.shape}")
print(f"  reward shape: {rew.shape}")

# Print first sample
print(f"\nFirst sequence, first step:")
print(f"  obs:    {obs[0, 0]}")
print(f"  action: {act[0, 0]}")
print(f"  reward: {rew[0, 0]:.2f}")

print("\n✅ Data loads correctly!")