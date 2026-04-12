import torch
import torch.nn as nn
import torchsde

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
        # Concatenate hidden state and control input
        input_tensor = torch.cat([h, x], dim=-1)
        return self.net(input_tensor)


class DiffusionNet(nn.Module):
    """Variance-preserving diffusion coefficient"""
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        # Scale for variance-preserving property
        self.scale = nn.Parameter(torch.ones(output_dim) * 0.1)
    
    def forward(self, h):
        return self.scale * torch.sigmoid(self.net(h))


class NSDEModel(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drift = DriftNet(feature_dim + hidden_dim, 128, hidden_dim)
        self.diffusion = DiffusionNet(hidden_dim, hidden_dim)
        
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)   # outputs: mu and log_sigma
        )

    def forward(self, x_path, t_span):
        """
        x_path: Tensor of shape (batch_size, time_steps, feature_dim)
        t_span: Tensor of integration times, e.g. torch.linspace(0, 1, steps)
        """
        batch_size = x_path.shape[0]
        device = x_path.device
        
        # Initial hidden state
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Define drift and diffusion functions for torchsde
        def drift(t, h):
            # Simple linear interpolation of control path for current time t
            idx = (t * (x_path.shape[1] - 1)).long().clamp(0, x_path.shape[1] - 1)
            x_t = x_path[:, idx]
            return self.drift(t, h, x_t)
        
        def diffusion(t, h):
            return self.diffusion(h)
        
        # Solve the SDE using torchsde
        hs = torchsde.sdeint(
            drift,
            diffusion,
            h0,
            t_span,
            method='euler',          # or 'milstein', 'stratonovich'
            dt=0.05,
            adaptive=False
        )
        
        # Take the final hidden state
        h_final = hs[-1]
        
        # Readout → mu and log_sigma
        out = self.readout(h_final)
        mu = out[:, 0]
        log_sigma = out[:, 1]
        
        return mu, log_sigma
