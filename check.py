import torch
import mujoco
import custom_envs
import gymnasium as gym

env = gym.make("SberInvertedPendulum-v0", max_episode_steps = 5000, render = False, spawn_ball_every = 1000)
env.reset()
for i in range(1200):
    state, reward, terminated, truncated, _ = env.step(0.0)
    print("returns",i, state, reward, terminated, truncated, _)
    done = terminated or truncated
    if done:
        env.reset()
        

