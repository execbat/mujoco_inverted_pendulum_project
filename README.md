# MuJoCo Inverted Pendulum Target

This project provides a **custom MuJoCo environment** where an agent controls an inverted pendulum.  
The task is to **move the pole tip to a randomly spawned target** and **hold balance** near the target point.
Mass of the pole is randomly choosen in range [10;30] kg.

The environment is designed for **training with PPO (Proximal Policy Optimization)**.

---
```
mujoco_inverted_pendulum_project/
├── custom_envs/                # Custom gym environments (your environment code)
│   ├── __init__.py
│   └── nn_arch.py               # Neural network architecture
│
├── experiments/                 # Directory for experiment outputs
│
├── runs/                        # Directory for TensorBoard logs
│
├── utils/                       # Utility scripts
│   ├── __init__.py
│   └── nn_arch.py
│
│
├── config.yaml                  # Configuration file for environment and training
├── inverted_pendulum_env.py     # Base MuJoCo environment setup
├── inv_pendulum_gym_env_0.py    # Gym wrapper for custom environment
├── train_cpu.py                 # Script to train agent on CPU
├── train_gpu.py                 # Script to train agent on GPU
├── enjoy.py                     # Script to run the trained agent (demonstration)
├── functions_calc.ipynb         # Jupyter Notebook with speed functions and calculations
├── requirements.txt             # Python dependencies
├── .gitignore                   # Files to ignore in Git
└── README.md                    # Project description and instructions
```
---

## 🛠️ Installation

First, clone the repository:
```
bash

git clone https://github.com/execbat/mujoco_inverted_pendulum_project.git
cd mujoco_inverted_pendulum_project
```

```pip install -r requirements.txt```

✅ Make sure you have MuJoCo installed and working.


## ⚙️ Configuration (config.yaml)

Main configuration is located in `config.yaml`:

```
yaml

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

## 🚀 How to Run

Train an agent

Run PPO on CPU: `python train_cpu.py`

Run PPO on GPU: `python train_gpu.py`

Run APPO on GPU: `python train_appo_gpu.py`


## 🚀 How to track the learning performance with Tensorboard

`cd mujoco_inverted_pendulum_project` and then `tensorboard --logdir=runs`



If you want to continue training from a previous checkpoint: `python train_gpu.py --continue` or `python train_cpu.py --continue` or `python train_appo_gpu.py --continue`


## 🎮 Run the trained agent (Demonstration)

After training, you can run the agent using: `python enjoy.py`

## 🎯 Environment Description

    Task: Move the tip of the pole to a dynamically spawned ball target and stabilize. Mass of the pole is randomly choosen [10;30] kg.

    Observation Space:

        - Cart position and velocity       

        - Pole angle and angular velocity  
        
        - Arc distance between the pole end point and vertical point above the cart

        - Linear distance between the pole end point and target point
        
        - X, Z coordinates of the pole end point in World CS
        
        - X, Z coordinates of the target point in World CS 
        
        - Mass of the pole
        
        - Gravity moment of the pole            

        - Linear velocity of the pole end point
        
        - Acceleration of the pole end point


    Action Space:

        - Continuous control (horizontal force applied to the cart)
        

The environment rewards:

    Matching the pole end point velocity with manually designed velocity function. (Imitation learning)
    
    Matching the pole end point acceleration with acceleration function derived from manually designed velocity function. (Imitation learning)

    Keeping the pole upwards
    
    Compensation of the gravity moment when the pole stays upwards

    Keeping the pole end point closer to the target point with minimal velocity

    Penalizes:

        Leaving the screen bounds

        Inactivity (no changes between steps)
        

## 📋 Requirements:

```
gymnasium
numpy
matplotlib
torch
ruamel.yaml
mujoco
mujoco-py
```







