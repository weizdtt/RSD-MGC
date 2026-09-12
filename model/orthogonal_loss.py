import torch
import torch.nn as nn
import torch.nn.functional as F
from model.LSTM_MU_block import IterativeUnit


class DualOrthogonalNet(nn.Module):
    def __init__(self, feature_dim=256, n_views=3, channels=8, reduce_dim=512, iters=5):
        super().__init__()
        self.feature_dim = feature_dim

        self.bi_branch = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.LayerNorm(1024),
            nn.Conv1d(n_views, channels, kernel_size=4, padding='same'),
            nn.AdaptiveAvgPool1d(reduce_dim),
            nn.Linear(reduce_dim, 1024),
            nn.SiLU(),
            nn.Conv1d(channels, n_views, kernel_size=4, padding='same'),
            IterativeUnit(1024, iters),
            nn.Linear(1024, 2*feature_dim),
            nn.LayerNorm(2*feature_dim),
        )

        self.ortho_lambda = nn.Parameter(torch.tensor(0.3))

    def forward(self, x):
        feature_dim = x.shape[-1]
        x = x.permute(1, 0, 2)
        features = self.bi_branch(x)
        features = features.permute(1, 0, 2)
        main = features[:, :, : feature_dim]
        secondary = features[:, :, feature_dim:]
        return main, secondary

    def orthogonal_loss(self, main, secondary):
        """
        shape: [n_views, batch_size, feature_dim]
        """
        main_norm = F.normalize(main, p=2, dim=-1)  # [n_views, bs, dim]
        sec_norm = F.normalize(secondary, p=2, dim=-1)

        diag_terms = []
        for v in range(main.shape[0]):
            similarity = torch.mm(main_norm[v], sec_norm[v].t())  # [bs, bs]
            diag_terms.append(torch.diag(similarity))  # [bs]
        diag_terms = torch.stack(diag_terms, dim=1)  # [bs, n_views]

        return self.ortho_lambda * torch.mean(torch.abs(diag_terms))

