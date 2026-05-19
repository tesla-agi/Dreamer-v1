import gymnasium as gym

env=gym.make('Pendulum-v1')

obs,info=env.reset(seed=42)

print("Starting Observation",obs)
print("10 random actions")

for step in range(10):
    action=env.action_space.sample()
    obs,reward,terminated,truncated,info=env.step(action)
    print(f"Step {step}: action={action[0]:+.2f} | "
          f"obs=[{obs[0]:+.2f}, {obs[1]:+.2f}, {obs[2]:+.2f}] | "
          f"reward={reward:.2f}")

env.close()
print("Done")