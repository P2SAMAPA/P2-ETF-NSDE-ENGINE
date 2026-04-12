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
        self.scale = nn.Parameter(torch.ones(output_dim) * 0.1)
    
    def forward(self, h):
        return self.scale * torch.sigmoid(self.net(h))


class NSDE_SDE(torch.nn.Module):
    """SDE wrapper for torchsde"""
    noise_type = "diagonal"   # required by torchsde
    sde_type = "ito"          # required by torchsde

    def __init__(self, drift_net, diffusion_net, x_path, t_span):
        super().__init__()
        self.drift_net = drift_net
        self.diffusion_net = diffusion_net
        self.x_path = x_path          # (batch, time_steps, feature_dim)
        self.t_span = t_span          # (time_steps,)

    def f(self, t, h):
        # Interpolate control path at time t
        # t is a scalar (or 0-dim tensor) but can be batched; we assume t in [0, t_span[-1]]
        t_norm = t / self.t_span[-1]  # map to [0,1]
        idx = (t_norm * (self.x_path.shape[1] - 1)).long().clamp(0, self.x_path.shape[1] - 1)
        x_t = self.x_path[:, idx, :]  # (batch, feature_dim)
        return self.drift_net(t, h, x_t)

    def g(self, t, h):
        return self.diffusion_net(h)


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
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        
        # Create the SDE object
        sde = NSDE_SDE(self.drift, self.diffusion, x_path, t_span)
        
        # Solve the SDE
        hs = torchsde.sdeint(
            sde,
            h0,
            t_span,
            method='euler',
            dt=0.05,
            adaptive=False,
            options={'dt': 0.05}
        )  # shape: (len(t_span), batch, hidden_dim)
        
        h_final = hs[-1]  # (batch, hidden_dim)
        out = self.readout(h_final)
        mu = out[:, 0]
        log_sigma = out[:, 1]
        return mu, log_sigma
