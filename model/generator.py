import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class MPSiLU(nn.Module):
    def forward(self, x):
        return F.silu(x) / 0.596


class Generator(nn.Module):
    def __init__(self, dim):
        super(Generator, self).__init__()
        self.networks = nn.Sequential(nn.Linear(dim, 1024, bias=True),
                                      nn.BatchNorm1d(1024),
                                      nn.Linear(1024, 1024, bias=False),
                                      nn.Linear(1024, dim, bias=True),
                                      )

        self.z_proj = nn.Sequential(
            nn.Linear(dim*2, dim*2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(2*dim, dim, bias=True),
        )

        self.time_embed = nn.Sequential(
            nn.Linear(dim, 3*dim),
            nn.SiLU(),
            nn.Linear(3*dim, dim),
        )

        self.dim = dim

    def forward(self, z_t, x, time_steps, self_cond=None):
        z_t = z_t.to(x.dtype)
        self_cond = self.default(self_cond, lambda: torch.zeros_like(z_t))
        z_t = torch.cat((self_cond, z_t), dim=-1)
        z_t = self.z_proj(z_t)
        time_token = self.time_embed(self.timestep_embedding(time_steps))
        return self.networks(x + z_t + time_token)

    @staticmethod
    def exists(x):
        return x is not None

    def default(self, val, d):
        if self.exists(val):
            return val
        return d() if callable(d) else d

    def timestep_embedding(self, time_steps, max_period=10000):
        """
        Create sinusoidal timestep embeddings.

        :param time_steps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=time_steps.device)
        args = time_steps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
