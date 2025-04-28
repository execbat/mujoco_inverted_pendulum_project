# UPDATED: PPO training with GPU support
from utils import Actor, Critic
import numpy as np
import gymnasium as gym
import mujoco
import custom_envs
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import yaml
import argparse
import os
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_latest_model(folder_path, key = "actor"):
    files = [f for f in os.listdir(folder_path) if key in f and f.endswith(".pt")]
    files = [os.path.join(folder_path, f) for f in files if os.path.isfile(os.path.join(folder_path, f))]
    if not files:
        return None
    return max(files, key=os.path.getctime)

def compute_gae(rewards, values, dones, gamma, lam):
    returns = []
    gae = 0
    values = values + [torch.tensor(0.0, device=device)]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * (1.0 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1.0 - dones[i]) * gae
        returns.insert(0, gae + values[i])
    return returns

def collect_samples(env, actor, critic, gamma, lam):
    states, actions, log_probs, dones, mus, stds = [], [], [], [], [], []
    rewards, values = [], []

    state, _ = env.reset()
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        states.append(state_tensor)

        mu, std = actor(state_tensor.unsqueeze(0))
        mus.append(mu.detach().squeeze(0))
        stds.append(std.detach().squeeze(0))

        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        #action_clipped = torch.clamp(action, -3.0, 3.0)

        log_prob = dist.log_prob(action).sum(axis=-1)
        log_probs.append(log_prob)

        value = critic(state_tensor.unsqueeze(0)).squeeze(0)
        values.append(value)

        next_state, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated

        rewards.append(reward)
        dones.append(done)
        actions.append(action.squeeze(0))

        state = next_state

    total_reward = sum(rewards)
    returns = compute_gae(rewards, values, dones, gamma, lam)
    advantages = torch.tensor(returns, dtype=torch.float32, device=device) - torch.stack(values)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return (
        torch.stack(states),
        torch.stack(actions),
        torch.stack(log_probs).detach(),
        torch.tensor(returns, dtype=torch.float32, device=device),
        advantages.detach(),
        total_reward,
        torch.stack(mus),
        torch.stack(stds),
    )

def train(env, actor, critic, gamma=0.99, lam=0.95, clip_eps=0.2, lr=0.001,
          entropy_coef=0.001, episodes=1000, update_epochs=5, batch_size=64,
          print_reward_every=10, log_dir="runs/ppo_run/", save_model_every=10,
          kl_treshold=0.03):

    actor_optim = optim.Adam(actor.parameters(), lr=lr)
    critic_optim = optim.Adam(critic.parameters(), lr=lr)
    reward_collection = []
    writer = SummaryWriter(log_dir=log_dir)
    kl_spike_counter = 0

    for episode in range(episodes):
        max_reward_ever = float('-inf')
        states, actions, old_log_probs, returns, advantages, total_reward, old_mus, old_stds = collect_samples(env, actor, critic, gamma, lam)
        reward_collection.append(total_reward)
        max_reward_ever = max(max_reward_ever, total_reward)

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
                clip_eps = max(clip_eps * 0.9, 0.05)
            if kl_div < kl_treshold * 0.5:
                clip_eps = min(clip_eps * 1.1, 0.3)
            writer.add_scalar("clip_eps", clip_eps, episode)

            if sum(reward_collection[-3:]) / 3.0 < max_reward_ever * 0.9 and episode > 10:
                kl_spike_counter += 1
            else:
                kl_spike_counter = max(0, kl_spike_counter - 1)
            writer.add_scalar("kl_spike_counter", kl_spike_counter, episode)

            if False: #kl_spike_counter >= update_epochs * 20:
                print("Early stopping update due to high KL divergence")
                print("kl_div", kl_div.item())
                torch.save(actor, os.path.join(save_dir, f"actor_{episode+1}.pt"))
                torch.save(critic, os.path.join(save_dir, f"critic_{episode+1}.pt"))
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
            print(f"Episode {episode+1}, mean reward {sum(reward_collection[-10:]) / 10:.2f}")

        if (episode + 1) % save_model_every == 0:
            torch.save(actor, os.path.join(save_dir, f"actor_{episode+1}.pt"))
            torch.save(critic, os.path.join(save_dir, f"critic_{episode+1}.pt"))
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

    env = gym.make(env_name, max_episode_steps=5000, render=False, spawn_ball_every=1000)
    n_obs = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    save_dir = os.path.join("experiments", experiment_name)
    os.makedirs(save_dir, exist_ok=True)

    if os.path.exists(save_dir) and args.cont:
        actor = torch.load(get_latest_model(save_dir, "actor"), map_location=device, weights_only = False)
        critic = torch.load(get_latest_model(save_dir, "critic"), map_location=device, weights_only = False)
        actor.train()
        critic.train()
        print("Models were loaded successfully")  
    else:
        actor = Actor(n_obs, n_actions).to(device)
        critic = Critic(n_obs).to(device)
        print("Models has been created")

    print("training started")
    train(env, actor, critic,
          gamma=ppo_params["gamma"],
          lam=ppo_params["lam"],
          clip_eps=ppo_params["clip_eps"],
          lr=ppo_params["lr"],
          entropy_coef=ppo_params["entropy_coef"],
          episodes=ppo_params["episodes"],
          update_epochs=ppo_params["update_epochs"],
          batch_size=ppo_params["batch_size"],
          print_reward_every=ppo_params["print_reward_every"],
          log_dir=log_params["log_dir"],
          save_model_every=checkpoint_params["save_model_every"],
          kl_treshold=ppo_params["kl_treshold"])
    print("training finished")

