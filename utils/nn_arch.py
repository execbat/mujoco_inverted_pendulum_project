import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, n_obs, n_actions):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(32, n_actions)
        self.log_std = nn.Parameter(torch.zeros(n_actions))  
        
        # init weights and bias
        nn.init.uniform_(self.mu_head.weight, -0.003, 0.003)
        nn.init.constant_(self.mu_head.bias, 0.0)
        
    def forward(self, x):
        x = self.fc(x)
        mu = self.mu_head(x)
        std = self.log_std.exp().expand_as(mu)  
        return mu, std

class Critic(nn.Module):
    def __init__(self, n_obs):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)
