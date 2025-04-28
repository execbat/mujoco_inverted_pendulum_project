# MuJoCo Inverted Pendulum Target

This project provides a **custom MuJoCo environment** where an agent controls an inverted pendulum.  
The task is to **move the pole tip to a randomly spawned target** and **hold balance** near the target point.

The environment is designed for **training with PPO (Proximal Policy Optimization)**.

---

## 📂 Project Structure
custom_envs/ 
# (Optional) Custom environments experiments/ 
# Saved training results and models runs/ 
# TensorBoard logs utils/ 
# Neural network architecture (nn_arch.py) config.yaml 
# Training configuration  
# Training script for CPU train_cpu.py
# Training script for GPU train_gpu.py 
# Demonstration (inference) script enjoy.py 
# Base simulation environment inv_pendulum_gym_env_0.py 
# Gym-compatible 
custom environment  
# Utilities 

---

## 🛠️ Installation

First, clone the repository:

```bash

git clone https://github.com/execbat/mujoco_inverted_pendulum_project.git
cd mujoco_inverted_pendulum_project```


```pip install -r requirements.txt```
✅ Make sure you have MuJoCo installed and working.

⚙️ Configuration (config.yaml)
Main configuration is located in `config.yaml`:
```
env_name: "SberInvertedPendulum-v0"
experiment_name: "experiment_0"

ppo_params:
  gamma: 0.99
  lam: 0.95
  clip_eps_start: 0.2
  clip_eps_max: 0.1
  clip_eps_min: 0.05
  lr: 0.000003
  entropy_coef: 0.0001
  episodes: 100000
  update_epochs: 2
  batch_size: 512
  print_reward_every: 10
  max_episode_steps: 2000
  kl_treshold: 0.01

log_params:
  log_dir: "runs/ppo_run"

checkpoint_params:
  save_model_every: 50
```

🚀 How to Run
Train an agent

For GPU: `python train_gpu.py`
For CPU: `python train_cpu.py`
If you want to continue training from a previous checkpoint: `python train_gpu.py --continue` or `python train_cpu.py --continue`

🎮 Run the trained agent (Demonstration)
After training, you can run the agent using: `python enjoy.py`

🎯 Environment Description

    Task: Move the tip of the pole to a dynamically spawned ball target and stabilize. Mass of the pole is randomly choosen [10;30] kg.

    Observation Space:

        Cart position and velocity       

        Pole angle and angular velocity  

        Distance to target               

        Tip velocity

        Potential energy

        Tip coordinates

    Action Space:

        Continuous control (horizontal force applied to the cart)

The environment rewards:

    Approaching the vertical point smoothly

    Proper tip speed

    Smooth acceleration

    Staying vertical

    Reaching and holding near the target

    Penalizes:

        Leaving the screen bounds

        High accelerations if too close to the target

        Inactivity (no changes between steps)

📋 Requirements:
```
gymnasium
numpy
matplotlib
torch
ruamel.yaml
mujoco
mujoco-py
```







