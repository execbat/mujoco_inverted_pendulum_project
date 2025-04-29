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

import os

os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_latest_model(folder_path, key="actor"):
    files = [f for f in os.listdir(folder_path) if key in f and f.endswith(".pt")]
    if not files:
        return None
    return max([os.path.join(folder_path, f) for f in files], key=os.path.getctime)

def make_env(env_name):
    def _init():
        env = gym.make(env_name, max_episode_steps=5000, render=False, spawn_ball_every=1000)
        return env
    return _init

def compute_gae(rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """Compute GAE advantages in a vectorized way."""
    # Append last value for bootstrap
    values = torch.cat([values, torch.zeros(1, device=values.device)])
    deltas = rewards + gamma * values[1:] * (1 - dones) - values[:-1]

    advantages = torch.zeros_like(deltas)
    gae = 0
    for t in reversed(range(len(deltas))):
        gae = deltas[t] + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages
    
def collect_samples_parallel(envs, actor, critic, gamma, lam, steps_per_env= 2000):
    """Collect samples in parallel environments."""
    states, actions, log_probs, rewards, dones, values, mus, stds = [], [], [], [], [], [], [], []

    # Reset all envs
    state, _ = envs.reset()
    state = np.array(state)
    
     
    for _ in range(steps_per_env):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        
        # Get action distributions
        mu, std = actor(state_tensor)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(axis=-1)

        # Step environments
        next_state, reward, done, truncated, _ = envs.step(action.cpu().numpy())
        next_state = np.array(next_state)
        
        # Store collected data
        states.append(state_tensor)
        actions.append(action)
        log_probs.append(log_prob.detach())
        rewards.append(torch.tensor(reward, dtype=torch.float32, device=device))
        dones.append(torch.tensor(done, dtype=torch.float32, device=device))
        mus.append(mu.detach())
        stds.append(std.detach())

        # Get value estimation
        value = critic(state_tensor).squeeze(-1)
        values.append(value.detach())

        # Reset done environments
        if np.any(done):
            reset_indices = np.where(done)[0]
            for idx in reset_indices:
                reset_obs, _ = envs.reset_done()
                next_state[done] = reset_obs[done]

        state = next_state
    
    
    # Compute advantages
    values = torch.stack(values + [torch.zeros_like(values[0])])  # Append bootstrap value
    rewards = torch.stack(rewards)
    dones = torch.stack(dones)

    
    returns = []
    gae = 0
    for i in reversed(range(steps_per_env)):
        delta = rewards[i] + gamma * values[i+1] * (1.0 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1.0 - dones[i]) * gae
        returns.insert(0, gae + values[i])

    returns = torch.stack(returns)
    advantages = returns - values[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    min_size = min([len(i) for i in [states, returns, advantages]])
    
    states = torch.stack(states[:min_size])
    actions = torch.stack(actions[:min_size])
    log_probs = torch.stack(log_probs[:min_size])
    returns = returns[:min_size]
    advantages = advantages[:min_size].detach()
    mus = torch.stack(mus[:min_size])
    stds = torch.stack(stds[:min_size])
    
    total_reward = rewards.cpu().numpy().sum()
    return states, actions, log_probs, returns, advantages, mus, stds, total_reward



def train(envs: gym.vector.AsyncVectorEnv,
          actor: nn.Module,
          critic: nn.Module,
          gamma: float = 0.99,
          lam: float = 0.95,
          clip_eps: float = 0.2,
          lr: float = 0.001,
          entropy_coef: float = 0.001,
          episodes: int = 1000,
          update_epochs: int = 5,
          batch_size: int = 512,
          steps_per_env: int = 128,
          log_dir: str = "runs/ppo_run/",
          save_model_every: int = 10,
          kl_treshold: float = 0.03) -> None:
    """
    Train PPO agent using parallel environments.

    Args:
        envs: Vectorized parallel environments.
        actor: Policy network.
        critic: Value network.
        gamma: Discount factor.
        lam: GAE lambda parameter.
        clip_eps: Clipping epsilon for PPO.
        lr: Learning rate.
        entropy_coef: Entropy coefficient for exploration bonus.
        episodes: Number of training episodes.
        update_epochs: Number of epochs per training step.
        batch_size: Minibatch size.
        steps_per_env: Steps collected per environment before update.
        log_dir: Directory for TensorBoard logs.
        save_model_every: Frequency (episodes) to save models.
        kl_treshold: KL-divergence threshold for adaptive clip adjustment.
    """
    actor_optim = optim.Adam(actor.parameters(), lr=lr)
    critic_optim = optim.Adam(critic.parameters(), lr=lr)
    writer = SummaryWriter(log_dir=log_dir)

    reward_history = []
    clip_eps_start = clip_eps

    for episode in range(episodes):
        states, actions, old_log_probs, returns, advantages, mus, stds, total_reward = collect_samples_parallel(envs, actor, critic, gamma, lam, steps_per_env)

        idx = torch.randperm(states.shape[0])        
        reward_history.append(total_reward)        

        for _ in range(update_epochs):
            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_old_log_probs = old_log_probs[idx]
            batch_returns = returns[idx]
            batch_advantages = advantages[idx]
            batch_mus = mus[idx]
            batch_stds = stds[idx]

            mu, std = actor(batch_states)
            dist = torch.distributions.Normal(mu, std)
            new_log_probs = dist.log_prob(batch_actions).sum(axis=-1)
            entropy = dist.entropy().sum(axis=-1).mean()

            # KL divergence for adaptive clip_eps adjustment
            with torch.no_grad():
                old_dist = torch.distributions.Normal(batch_mus, batch_stds)
                kl_div = torch.distributions.kl_divergence(old_dist, dist).sum(axis=-1).mean()

            if kl_div > kl_treshold * 1.5:
                clip_eps = max(clip_eps * 0.9, 0.05)
            elif kl_div < kl_treshold * 0.5:
                clip_eps = min(clip_eps * 1.1, 0.3)

            # Loss calculation
            ratio = (new_log_probs - batch_old_log_probs).exp()
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean() - entropy_coef * entropy
            critic_loss = nn.MSELoss()(critic(batch_states).squeeze(-1), batch_returns)

            # Backward pass
            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()



        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(reward_history[-10:])
            writer.add_scalar("Rewards/Avg_Reward_10", avg_reward, episode)
            print(f"Episode {episode+1}: Avg reward (last 10) = {avg_reward:.2f} | Current clip_eps = {clip_eps:.4f}")

        if (episode + 1) % save_model_every == 0:
            torch.save(actor, os.path.join(save_dir, f"actor_{episode+1}.pt"))
            torch.save(critic, os.path.join(save_dir, f"critic_{episode+1}.pt"))
            print(f"Saved models at episode {episode+1}")
            
        # Logging to TensorBoard
        writer.add_scalar("Loss/Actor", actor_loss.item(), episode)
        writer.add_scalar("Loss/Critic", critic_loss.item(), episode)
        writer.add_scalar("Metrics/KL_Div", kl_div.item(), episode)
        writer.add_scalar("Metrics/Entropy", entropy.item(), episode)
        writer.add_scalar("Metrics/Clip_eps", clip_eps, episode)     


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

    num_envs = 32  # <- Сколько одновременно окружений запускаем
    save_dir = os.path.join("experiments", experiment_name)
    

    envs = gym.vector.AsyncVectorEnv([make_env(env_name) for _ in range(num_envs)])
    n_obs = envs.single_observation_space.shape[0]
    n_actions = envs.single_action_space.shape[0]

    if os.path.exists(save_dir):
        if not args.cont:
            print("This experiment already exists. Use '--continue' to resume training.")
            exit(0)
        actor = torch.load(get_latest_model(save_dir, "actor"), map_location=device, weights_only = False)
        critic = torch.load(get_latest_model(save_dir, "critic"), map_location=device, weights_only = False)
        actor.train()
        critic.train()
        print("Models were loaded successfully")  
    else:    
        os.makedirs(save_dir, exist_ok=True)
        actor = Actor(n_obs, n_actions).to(device)
        critic = Critic(n_obs).to(device)
        print("Models have been created")

    print("Training started")
    train(envs, actor, critic,
          gamma=ppo_params["gamma"],
          lam=ppo_params["lam"],
          clip_eps=ppo_params["clip_eps_start"],
          lr=ppo_params["lr"],
          entropy_coef=ppo_params["entropy_coef"],
          episodes=ppo_params["episodes"],
          update_epochs=ppo_params["update_epochs"],
          batch_size=ppo_params["batch_size"],
          steps_per_env=128,  # <- Сколько шагов на одно окружение между апдейтами
          log_dir=log_params["log_dir"],
          save_model_every=checkpoint_params["save_model_every"],
          kl_treshold=ppo_params["kl_treshold"])
    print("Training finished")

