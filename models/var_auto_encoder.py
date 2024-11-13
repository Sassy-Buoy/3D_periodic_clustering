"""Variational AutoEncoder class."""

import torch
from torch import nn, Tensor
from models.auto_encoder import AutoEncoder


class VarAutoEncoder(AutoEncoder):
    """Autoencoder class that inherits from PyTorch's nn.Module class."""

    def __init__(
        self,
        kernels: list[list[int]],
        batch_norm: bool = True,
        activation: str = "Relu",
        latent_dim: int = 96,
    ):
        super(VarAutoEncoder, self).__init__(
            kernels, batch_norm, activation, latent_dim
        )
        self.fc_mean = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(self.latent_dim, self.latent_dim)

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: Tensor) -> Tensor:
        """Encode the input."""
        x = self.encoder(x)
        mean, log_var = self.fc_mean(x), self.fc_log_var(x)
        return mean, log_var

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the autoencoder."""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar

    def get_recon_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Calculate the loss function."""
        criterion = nn.MSELoss()
        loss = criterion(x_recon, x)
        return loss

    def get_kl_divergence(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Calculate the KL divergence."""
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_divergence

    def latent_space(self, x: Tensor) -> Tensor:
        """Get the latent space representation."""
        mean, _ = self.encode(x)
        latent_space = mean.detach().cpu().numpy()
        latent_space = (latent_space - latent_space.mean(axis=0)) / latent_space.std(
            axis=0
        )
        return latent_space
