import os
import math
import random
import argparse
import yaml
from datetime import datetime

import numpy as np
import gymnasium as gym
import mujoco
import custom_envs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from utils import Actor, Critic


def get_latest_model(folder_path: str, key: str = "actor") -> str | None:
    """Find the latest model checkpoint in the given folder."""
    files = [f for f in os.listdir(folder_path) if key in f and f.endswith(".pt")]
    files = [os.path.join(folder_path, f) for f in files if os.path.isfile(os.path.join(folder_path, f))]

    if not files:
        return None
    return max(files, key=os.path.getctime)


def compute_gae(rewards, values, dones, gamma: float, lam: float):
    """Compute Generalized Advantage Estimation (GAE)."""
    returns = []
    gae = 0
    values = values + [torch.tensor(0.0)]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * (1.0 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1.0 - dones[i]) * gae
        returns.insert(0, gae + values[i])
    return returns


def collect_samples(env, actor, critic, gamma: float, lam: float):
    """Collect rollout samples from the environment."""
    states, actions, log_probs, dones, mus, stds = [], [], [], [], [], []
    rewards, values = [], []

    state, _ = env.reset()
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        states.append(state_tensor)

        mu, std = actor(state_tensor.unsqueeze(0))
        mus.append(mu.detach().squeeze(0))
        stds.append(std.detach().squeeze(0))

        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        log_probs.append(log_prob)

        value = critic(state_tensor.unsqueeze(0)).squeeze(0)
        values.append(value)

        next_state, reward, terminated, truncated, _ = env.step(action.squeeze(0).detach().numpy())
        done = terminated or truncated

        rewards.append(reward)
        dones.append(done)
        actions.append(action.squeeze(0))

        state = next_state

    total_reward = sum(rewards)
    returns = compute_gae(rewards, values, dones, gamma, lam)
    advantages = torch.tensor(returns, dtype=torch.float32) - torch.tensor(values, dtype=torch.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return (
        torch.stack(states),
        torch.stack(actions),
        torch.stack(log_probs).detach(),
        torch.tensor(returns, dtype=torch.float32),
        advantages.detach(),
        total_reward,
        torch.stack(mus),
        torch.stack(stds)
    )


def train(env,
          actor,
          critic,
          gamma: float = 0.99,
          lam: float = 0.95,
          clip_eps_start: float = 0.2,
          clip_eps_max: float = 0.2,
          clip_eps_min: float = 0.05,
          lr: float = 0.001,
          entropy_coef: float = 0.001,
          episodes: int = 1000,
          update_epochs: int = 5,
          batch_size: int = 64,
          print_reward_every: int = 10,
          log_dir: str = "runs/ppo_run/",
          save_model_every: int = 10,
          kl_treshold: float = 0.03):
    """Train PPO agent."""
    actor_optim = optim.Adam(actor.parameters(), lr=lr)
    critic_optim = optim.Adam(critic.parameters(), lr=lr)

    reward_collection = []
    writer = SummaryWriter(log_dir=log_dir)

    clip_eps = clip_eps_start
    kl_spike_counter = 0
    max_reward_ever = float('-inf')

    for episode in range(episodes):
        states, actions, old_log_probs, returns, advantages, total_reward, old_mus, old_stds = collect_samples(env, actor, critic, gamma, lam)
        reward_collection.append(total_reward)

        if max_reward_ever < total_reward:
            kl_spike_counter = 0
            max_reward_ever = total_reward

        writer.add_scalar("Rewards/TotalReward", total_reward, episode)

        for _ in range(update_epochs):
            idx = torch.randperm(states.shape[0])

            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_old_log_probs = old_log_probs[idx]
            batch_returns = returns[idx]
            batch_advantages = advantages[idx]
            batch_old_mus = old_mus[idx]
            batch_old_stds = old_stds[idx]

            mu, std = actor(batch_states)
            dist = torch.distributions.Normal(mu, std)
            new_log_probs = dist.log_prob(batch_actions).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1).mean()

            with torch.no_grad():
                old_dist = torch.distributions.Normal(batch_old_mus, batch_old_stds)
                kl_div = torch.distributions.kl_divergence(old_dist, dist).sum(axis=-1).mean()

            writer.add_scalar("kl_div", kl_div, episode)

            if kl_div > kl_treshold * 1.5:
                clip_eps = max(clip_eps * 0.9, clip_eps_min)
            if kl_div < kl_treshold * 0.5:
                clip_eps = min(clip_eps * 1.1, clip_eps_max)

            writer.add_scalar("clip_eps", clip_eps, episode)

            if sum(reward_collection[-3:]) / 3.0 < max_reward_ever * 0.9 and episode > 10:
                kl_spike_counter += 1
            else:
                kl_spike_counter = max(0, kl_spike_counter - 1)

            writer.add_scalar("kl_spike_counter", kl_spike_counter, episode)

            if kl_spike_counter >= update_epochs * 2 and False:
                print(f"Early stopping update due to high KL divergence: {kl_div:.4f}")
                actor_path = os.path.join(save_dir, f"actor_{episode+1}.pt")
                critic_path = os.path.join(save_dir, f"critic_{episode+1}.pt")
                torch.save(actor, actor_path)
                torch.save(critic, critic_path)
                return

            ratio = (new_log_probs - batch_old_log_probs).exp()
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy

            critic_loss = nn.MSELoss()(critic(batch_states).squeeze(-1), batch_returns)

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()

        if (episode + 1) % print_reward_every == 0:
            mean_reward = sum(reward_collection[-10:]) / 10
            print(f"Episode {episode+1}, mean reward {mean_reward:.2f}")

        if (episode + 1) % save_model_every == 0:
            actor_path = os.path.join(save_dir, f"actor_{episode+1}.pt")
            critic_path = os.path.join(save_dir, f"critic_{episode+1}.pt")
            torch.save(actor, actor_path)
            torch.save(critic, critic_path)
            print(f"Saved checkpoints at episode {episode+1}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--continue', dest='cont', action='store_true', help='Continue training from checkpoint')
    args = parser.parse_args()

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    ppo_params = config['ppo_params']
    log_params = config['log_params']
    env_name = config['env_name']
    checkpoint_params = config['checkpoint_params']
    experiment_name = config['experiment_name']

    env = gym.make(env_name, max_episode_steps=2000, render=False, spawn_ball_every=500)
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    experiments_dir = "experiments"
    os.makedirs(experiments_dir, exist_ok=True)
    save_dir = os.path.join(experiments_dir, experiment_name)

    if os.path.exists(save_dir):
        if not args.cont:
            print("This experiment already exists. Use '--continue' to resume training.")
            exit(0)

        actor = torch.load(get_latest_model(save_dir, "actor"), weights_only = False)
        critic = torch.load(get_latest_model(save_dir, "critic"), weights_only = False)
        actor.train()
        critic.train()
        print("Models loaded successfully.")
    else:
        os.makedirs(save_dir)
        actor = Actor(n_obs, n_actions)
        critic = Critic(n_obs)
        print("Models created.")

    print("Training started.")
    train(
        env,
        actor,
        critic,
        gamma=ppo_params["gamma"],
        lam=ppo_params["lam"],
        clip_eps_start=ppo_params["clip_eps_start"],
        clip_eps_max=ppo_params["clip_eps_max"],
        clip_eps_min=ppo_params["clip_eps_min"],
        lr=ppo_params["lr"],
        entropy_coef=ppo_params["entropy_coef"],
        episodes=ppo_params["episodes"],
        update_epochs=ppo_params["update_epochs"],
        batch_size=ppo_params["batch_size"],
        print_reward_every=ppo_params["print_reward_every"],
        log_dir=log_params["log_dir"],
        save_model_every=checkpoint_params["save_model_every"],
        kl_treshold=ppo_params["kl_treshold"]
    )
    print("Training finished.")

