import torch
import torch.nn as nn
from torchdiffeq import sdeint

class DriftNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, t, h, x):
        # Concat hidden state and control
        input = torch.cat([h, x], dim=-1)
        return self.net(input)

class DiffusionNet(nn.Module):
    """Variance-preserving diffusion coefficient"""
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        self.scale = nn.Parameter(torch.ones(output_dim) * 0.1)
    
    def forward(self, h):
        return self.scale * torch.sigmoid(self.net(h))

class NSDEModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim=64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drift = DriftNet(feature_dim + hidden_dim, 128, hidden_dim)
        self.diffusion = DiffusionNet(hidden_dim, hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)   # mu and log_sigma
        )
    
    def forward(self, x_path, t_span):
        """x_path: (batch, time, features)"""
        batch_size = x_path.shape[0]
        h0 = torch.zeros(batch_size, self.hidden_dim, device=x_path.device)
        
        def sde_drift(t, h):
            # Interpolate control at time t
            idx = (t * (x_path.shape[1]-1)).long().clamp(0, x_path.shape[1]-1)
            x_t = x_path[:, idx]
            return self.drift(t, h, x_t)
        
        def sde_diffusion(t, h):
            return self.diffusion(h)
        
        # Solve SDE
        hs = sdeint(sde_drift, sde_diffusion, h0, t_span, method='euler', dt=0.1)
        
        # Use final hidden state
        h_final = hs[-1]
        out = self.readout(h_final)
        mu, log_sigma = out[:, 0], out[:, 1]
        return mu, log_sigma
