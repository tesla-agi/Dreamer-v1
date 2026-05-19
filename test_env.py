import gymnasium as gym

env=gym.make('Pendulum-v1')
obs,info=env.reset(seed=42)

print("Observation:",obs)
print("Observation Shape",obs.shape)

env.close()