import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class AttentionLayer(nn.Module):
    def __init__(self, dim, n_views):
        super(AttentionLayer, self).__init__()
        self.dim = dim
        self.n_views = n_views
        self.mlp = nn.Sequential(
            nn.Linear(self.dim * self.n_views, self.dim * self.n_views),
            nn.BatchNorm1d(self.dim * self.n_views),
            nn.ReLU(),
            nn.Linear(self.dim * self.n_views, self.dim * self.n_views),
            nn.BatchNorm1d(self.dim * self.n_views),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(self.dim * self.n_views, self.n_views, bias=True)

    def forward(self, x, tau=10.0):
        h = x.permute(1, 0, 2).reshape(x.shape[1], -1)
        act = self.output_layer(self.mlp(h))
        act = F.sigmoid(act) / tau
        e = F.softmax(act, dim=1)
        hs = [e[:, v].unsqueeze(1) * x[v] for v in range(x.shape[0])]
        h = sum(hs)
        return h


class VanillaSelfAttention(nn.Module):
    def __init__(self, dim, n_views, num_heads=1, mode='sa_1', use_norm=True, p=3, dropout=0.5):
        super(VanillaSelfAttention, self).__init__()
        self.dim = dim
        self.mode = mode
        self.use_norm = use_norm
        self.p = p
        self.n_views = n_views
        self.fused = nn.Linear(dim * n_views, dim)
        self.self_attn = nn.TransformerEncoderLayer(
            d_model=dim*n_views, nhead=num_heads, dim_feedforward=256, dropout=dropout, batch_first=True) \
            if mode == 'sa_1' else nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        if mode == 'sa_2' or 'sa_3':
            self.norm1 = nn.LayerNorm(dim)
            self.norm2 = nn.LayerNorm(dim)
            self.dropout = nn.Dropout(dropout)
            self.dropout1 = nn.Dropout(dropout)
            self.dropout2 = nn.Dropout(dropout)
            self.linear1 = nn.Linear(dim, 256)
            self.linear2 = nn.Linear(256, dim)
            self.activation = nn.ReLU()

    def _normalize(self, x):
        if self.use_norm:
            return F.normalize(x, p=self.p, dim=-1)
        else:
            return x

    def _trans_encoder(self, x):
        x_c = x.mean(dim=1, keepdim=True).repeat(1, self.n_views, 1)
        x = self.norm1(self.dropout1(self.self_attn(x_c, x, x, need_weights=False)[0]).permute(1, 0, 2)[0] + x.mean(dim=1))
        return self.norm2(x + self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def _cross_attn(self, x):
        m_out = []
        x_m = x.mean(dim=1)
        _, views, _ = x.shape
        for i in range(views):
            x_c = (x_m - x[:, i, :] / views).unsqueeze(1)
            x_tmp = self.norm1(self.dropout1(self.self_attn(x_c, x[:, i, :].unsqueeze(1), x[:, i, :].unsqueeze(1), need_weights=False)[0]).squeeze(1) + x[:, i, :])
            m_out.append(self.norm2(x[:, i, :] + self.linear2(self.dropout(self.activation(self.linear1(x_tmp))))))
        return torch.stack(m_out, dim=1).mean(dim=1)

    def forward(self, x):
        """
        x: (view, batch, dim)
        """
        if self.mode == 'sa_1':
            x = x.permute(1, 0, 2).reshape(x.shape[1], -1)
            x = self.self_attn(x)
            return self._normalize(self.fused(x))
        elif self.mode == 'sa_2':
            x = x.permute(1, 0, 2)
            return self._normalize(self._cross_attn(x))
        elif self.mode == 'sa_3':
            x = x.permute(1, 0, 2)
            return self._normalize(self._trans_encoder(x))
        else:
            raise NotImplementedError


class AddLinearAttention(nn.Module):
    def __init__(self, dim, n_views, num_heads, mode='linear_only', batch_first=True, kv_mean=True):
        super().__init__()
        self.num_heads = num_heads
        self.mode = mode
        self.batch_first = batch_first
        self.kv_mean = kv_mean
        self.embed_dim = num_heads * n_views
        self.ex_ch = nn.Conv1d(n_views, self.embed_dim, kernel_size=1, padding='same') if num_heads > 1 else nn.Identity()
        if mode == 'linear_only':
            self.qkvo = nn.Conv1d(self.embed_dim, self.embed_dim*4, kernel_size=1, padding='same')
            self.lepe = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=4, padding='same')
            self.proj = nn.Conv1d(self.embed_dim, n_views, kernel_size=1, padding='same')
        else:
            self.qkvo = nn.Conv1d(self.embed_dim, self.embed_dim * 4, kernel_size=1, padding='same')
            self.lepe = nn.Conv1d(self.embed_dim, self.embed_dim, kernel_size=4, padding='same')
            self.proj = nn.Conv1d(self.embed_dim, 1, kernel_size=1, padding='same')
        self.scale = dim ** -0.5
        self.elu = nn.ELU()

    def forward(self, x: torch.Tensor):
        """"
        x: (batch, view, dim)
        q: (batch, view, dim)
        """
        batch, views, dim = x.shape
        if self.mode == 'linear_only':
            qkvo = self.qkvo(self.ex_ch(x))
            qkv = qkvo[:, :3 * self.embed_dim, :]
            o = qkvo[:, 3 * self.embed_dim:, :]
            lepe = self.lepe(qkv[:, 2 * self.embed_dim:, :])  # (b e d)

            if self.batch_first:
                q, k, v = rearrange(qkv, 'b (m n v) d -> m b n v d', m=3, n=self.num_heads)  # (b n v d)
                if self.kv_mean:
                    k = k.mean(dim=-2, keepdim=True).repeat(1, 1, views, 1)
                    v = v.mean(dim=-2, keepdim=True).repeat(1, 1, views, 1)
            else:
                q, k, v = rearrange(qkv, 'b (m n v) d -> m v n b d', m=3, n=self.num_heads)  # (v n b d)
                if self.kv_mean:
                    k = k.mean(dim=0, keepdim=True).repeat(views, 1, 1, 1)
                    v = v.mean(dim=0, keepdim=True).repeat(views, 1, 1, 1)

        elif self.mode == 'mixed':
            qkvo = self.qkvo(self.ex_ch(x))
            qkv = qkvo[:, :3 * self.embed_dim, :]
            o = qkvo[:, 3 * self.embed_dim:, :]
            lepe = self.lepe(qkv[:, 2 * self.embed_dim:, :])

            if self.batch_first:
                q, k, v = rearrange(qkv, 'b (m n v) d -> m b n v d', m=3, n=self.num_heads)  # (b n v d)
                if self.kv_mean:
                    k = k.mean(dim=-2, keepdim=True).repeat(1, 1, views, 1)
                    v = v.mean(dim=-2, keepdim=True).repeat(1, 1, views, 1)
            else:
                q, k, v = rearrange(qkv, 'b (m n v) d -> m v n b d', m=3, n=self.num_heads)  # (v n b d)
                if self.kv_mean:
                    k = k.mean(dim=0, keepdim=True).repeat(views, 1, 1, 1)
                    v = v.mean(dim=0, keepdim=True).repeat(views, 1, 1, 1)
        else:
            raise NotImplementedError

        q = self.elu(q) + 1
        k = self.elu(k) + 1
        v = self.elu(v) + 1

        z = q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) * self.scale

        l_scale = views if self.batch_first else batch

        kv = (k.transpose(-2, -1) * (self.scale / l_scale) ** 0.5) @ (v * (self.scale / l_scale) ** 0.5)

        res = q @ kv * (1 + 1 / (z + 1e-6)) - z * v.mean(dim=-2, keepdim=True)

        if self.batch_first:
            res = rearrange(res, 'b n v d -> b (n v) d', b=batch, n=self.num_heads, v=views, d=dim)
        else:
            res = rearrange(res, 'v n b d -> b (n v) d', b=batch, n=self.num_heads, v=views, d=dim)
        res = res + lepe
        return self.proj(res * o)


class MixedAttention(nn.Module):
    def __init__(self, dim, n_views, num_heads=1, mode_attn='mixed', mode_sfm='sa_1', use_norm=True, batch_first=True, kv_mean=True, p=3, dropout=0.5):
        super(MixedAttention, self).__init__()
        self.dim = dim
        self.n_views = n_views
        self.mode_attn = mode_attn
        if mode_attn == 'mixed':
            if mode_sfm == 'sa_4':
                self.sfm_attn = AttentionLayer(dim, n_views)
            else:
                self.sfm_attn = VanillaSelfAttention(dim, n_views, num_heads, mode_sfm, use_norm, p, dropout)
            self.lin_attn = AddLinearAttention(dim, n_views, num_heads, 'mixed', batch_first, kv_mean)
        elif mode_attn == 'softmax_only':
            self.sfm_attn = VanillaSelfAttention(dim, n_views, num_heads, mode_sfm, use_norm, p, dropout)
        elif mode_attn == 'weighted_only':
            self.w_attn = AttentionLayer(dim, n_views)
        elif mode_attn == 'linear_only':
            self.lin_attn = AddLinearAttention(dim, n_views, num_heads, 'linear_only', batch_first, kv_mean)
        else:
            raise NotImplementedError

    def forward(self, x):
        v, b, d = x.shape
        assert v == self.n_views and d == self.dim, 'shape of inputs mismatched parameters'
        if self.mode_attn == 'mixed':
            q = self.sfm_attn(x)
            q = q.unsqueeze(1).repeat(1, self.n_views, 1)
            x = self.lin_attn(q).permute(1, 0, 2)
            return x.repeat(self.n_views, 1, 1)
        elif self.mode_attn == 'softmax_only':
            x = self.sfm_attn(x)
            return x.unsqueeze(0).repeat(self.n_views, 1, 1)
        elif self.mode_attn == 'weighted_only':
            x = self.w_attn(x)
            return x.unsqueeze(0).repeat(self.n_views, 1, 1)
        elif self.mode_attn == 'linear_only':
            x = self.lin_attn(x.permute(1, 0, 2)).permute(1, 0, 2)
            return x


