import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module, ModuleList
from typing import Dict
import math

from dataclasses import dataclass

@dataclass(frozen=True)
class ModelMeta:
    name: str
    version: str
    description: str
    paper: str | None = None


# Modify model info here
META = ModelMeta(
    name="model_v1_cnn_t",
    version="1.0",
    description="CNN + Transformer baseline",
)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    logvar = logvar.clamp(min=-20, max=10)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mu + eps * std


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.size(1) #x: [B, T, D]
        return x + self.pe[:T].unsqueeze(0)  # [1, T, D]


class VAE(nn.Module):
    def __init__(
        self,
        n_features: int = 11,
        seq_len: int = 7,
        d_model: int = 64,
        cnn_channels: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        latent_dim: int = 16,
        dropout: float = 0.1,
        use_cls_token: bool = True,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_features = n_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.use_cls_token = use_cls_token

        # 1) Local feature extractor over time axis.
        # Input to conv1d should be [B, C_in, T]
        # 2-layer temporal conv.
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=n_features, out_channels=cnn_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=cnn_channels, out_channels=d_model, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # CLS token for pooling
        if use_cls_token:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.cls, std=0.02)

        self.posenc = SinusoidalPositionalEncoding(d_model=d_model, max_len=seq_len + (1 if use_cls_token else 0))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Latent head
        self.to_mu = nn.Linear(d_model, latent_dim)
        self.to_logvar = nn.Linear(d_model, latent_dim)

        # 2) Decoder: z -> sequence tokens -> refine -> project back to features
        self.z_to_tokens = nn.Sequential(
            nn.Linear(latent_dim, d_model * seq_len),
            nn.GELU(),
        )

        dec_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder_tf = nn.TransformerEncoder(dec_layer, num_layers=max(1, n_layers - 1))

        # Small conv head over time
        self.decoder_cnn = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=d_model, out_channels=n_features, kernel_size=1),
        )

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, F = x.shape
        assert T == self.seq_len and F == self.n_features, f"Expected [B,{self.seq_len},{self.n_features}] got {x.shape}"

        # CNN over time: [B, F, T] -> [B, d_model, T] -> [B, T, d_model]
        h = self.cnn(x.transpose(1, 2)).transpose(1, 2)

        if self.use_cls_token:
            cls = self.cls.expand(B, -1, -1)  # [B,1,D]
            h = torch.cat([cls, h], dim=1)    # [B,1+T,D]

        h = self.posenc(h)
        h = self.encoder(h)  # [B, 1+T, D] or [B, T, D]

        pooled = h[:, 0] if self.use_cls_token else h.mean(dim=1)  # [B, D]
        mu = self.to_mu(pooled)
        logvar = self.to_logvar(pooled)
        return {"mu": mu, "logvar": logvar, "enc": h}

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        B = z.size(0) #z: [B, latent_dim]
        tokens = self.z_to_tokens(z).view(B, self.seq_len, self.d_model)  # [B, T, D]
        tokens = self.posenc(tokens)
        tokens = self.decoder_tf(tokens)  # [B, T, D]

        # conv1d expects [B, D, T]
        x_hat = self.decoder_cnn(tokens.transpose(1, 2)).transpose(1, 2)  # [B, T, F]
        return x_hat

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encode(x)
        z = reparameterize(out["mu"], out["logvar"])
        x_hat = self.decode(z)
        out.update({"z": z, "x_hat": x_hat})
        return out
