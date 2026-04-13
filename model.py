import torch
import torch.nn as nn
import torchsde

class DriftNet(nn.Module):
    def __init__(self, hidden_dim, feature_dim, macro_dim):
        super().__init__()
        # Input: h (hidden_dim) + X (feature_dim) + M (macro_dim)
        input_dim = hidden_dim + feature_dim + macro_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim)
        )
    def forward(self, t, h, x, m):
        # x: (batch, feature_dim), m: (batch, macro_dim)
        inp = torch.cat([h, x, m], dim=-1)
        return self.net(inp)

class DiffusionNet(nn.Module):
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
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, drift_net, diffusion_net, x_path, macro_path, t_span):
        super().__init__()
        self.drift_net = drift_net
        self.diffusion_net = diffusion_net
        self.x_path = x_path          # (batch, steps, feat_dim)
        self.macro_path = macro_path  # (batch, steps, macro_dim)
        self.t_span = t_span

    def f(self, t, h):
        # Interpolate both X and M at time t
        t_norm = t / self.t_span[-1]
        idx = (t_norm * (self.x_path.shape[1] - 1)).long().clamp(0, self.x_path.shape[1] - 1)
        x_t = self.x_path[:, idx, :]
        m_t = self.macro_path[:, idx, :]
        return self.drift_net(t, h, x_t, m_t)

    def g(self, t, h):
        return self.diffusion_net(h)

class NSDEModel(nn.Module):
    def __init__(self, feature_dim: int, macro_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.drift = DriftNet(hidden_dim, feature_dim, macro_dim)
        self.diffusion = DiffusionNet(hidden_dim, hidden_dim)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x_path, macro_path, t_span):
        batch_size = x_path.shape[0]
        device = x_path.device
        h0 = torch.zeros(batch_size, self.hidden_dim, device=device)
        sde = NSDE_SDE(self.drift, self.diffusion, x_path, macro_path, t_span)
        hs = torchsde.sdeint(
            sde, h0, t_span,
            method='euler', dt=0.05, adaptive=False
        )
        h_final = hs[-1]
        out = self.readout(h_final)
        mu = out[:, 0]
        log_sigma = out[:, 1]
        return mu, log_sigma
