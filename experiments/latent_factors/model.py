from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Simple sparse autoencoder with linear bottleneck and L1 latent penalty."""

    def __init__(self, input_dim: int, latent_dim: int = 16, hidden_dim: int = 512, num_hidden: int = 2):
        super().__init__()
        encoder_layers = []
        dim = input_dim
        for _ in range(num_hidden):
            encoder_layers.append(nn.Linear(dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            dim = hidden_dim
        encoder_layers.append(nn.Linear(dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        dim = latent_dim
        for _ in range(num_hidden):
            decoder_layers.append(nn.Linear(dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            dim = hidden_dim
        decoder_layers.append(nn.Linear(dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


def sparse_ae_loss(recon: torch.Tensor, target: torch.Tensor, latent: torch.Tensor, sparsity_weight: float) -> dict:
    mse = F.mse_loss(recon, target)
    sparsity = torch.mean(torch.abs(latent))
    loss = mse + sparsity_weight * sparsity
    return {"loss": loss, "mse": mse.detach(), "sparsity": sparsity.detach()}
