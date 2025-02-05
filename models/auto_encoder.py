"""Autoencoder models for 3D image data."""

import torch
from torch import Tensor, nn

from models.conv_block import ConvBlock, ResBlock


class AutoEncoder(nn.Module):
    """Autoencoder class that inherits from PyTorch's nn.Module class."""

    def __init__(
        self,
        config: list[dict],
    ):
        super(AutoEncoder, self).__init__()
        self.config = config

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        for i, layer in enumerate(self.config):
            # Insert the encoder layers
            if layer["type"] == "Conv":
                self.encoder.append(
                    ConvBlock(
                        kernels=layer["kernel_size"],
                        in_channels=layer["in_channels"],
                        out_channels=layer["out_channels"],
                        in_size=layer["in_size"],
                        out_size=layer["out_size"],
                    )
                )
                # Insert the decoder layers in reverse order
                self.decoder.insert(
                    0,
                    ConvBlock(
                        kernels=layer["kernel_size"],
                        in_channels=layer["out_channels"],
                        out_channels=layer["in_channels"],
                        in_size=layer["out_size"],
                        out_size=layer["in_size"],
                    ),
                )
            elif layer["type"] == "Res":
                self.encoder.append(
                    ResBlock(
                        kernels=layer["kernel_size"],
                        in_channels=layer["in_channels"],
                        out_channels=layer["out_channels"],
                        in_size=layer["in_size"],
                        out_size=layer["out_size"],
                    )
                )
                self.decoder.insert(
                    0,
                    ResBlock(
                        kernels=layer["kernel_size"],
                        in_channels=layer["out_channels"],
                        out_channels=layer["in_channels"],
                        in_size=layer["out_size"],
                        out_size=layer["in_size"],
                    ),
                )
        # Flatten and unflatten the tensors
        self.encoder.append(nn.Flatten())
        self.decoder.insert(
            0, nn.Unflatten(1, (self.config[-1]["out_channels"], 1, 1, 1))
        )

    def forward(self, x: Tensor):
        """Forward pass through the autoencoder."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Calculate the loss function."""
        # criterion = CustomLoss()
        criterion = nn.MSELoss()
        return criterion(x_recon, x)


class VarAutoEncoder(AutoEncoder):
    """Autoencoder class that inherits from PyTorch's nn.Module class."""

    def __init__(self, config: list[dict]):
        super(VarAutoEncoder, self).__init__(config)
        self.fc_mean = nn.Linear(
            self.config[-1]["out_channels"], self.config[-1]["out_channels"]
        )
        self.fc_log_var = nn.Linear(
            self.config[-1]["out_channels"], self.config[-1]["out_channels"]
        )

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode the input."""
        x = self.encoder(x)
        mean, log_var = self.fc_mean(x), self.fc_log_var(x)
        return mean, log_var

    def forward(self, x: Tensor):
        """Forward pass through the autoencoder."""
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_recon = self.decoder(z)
        return x_recon, mean, logvar

    def get_recon_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Calculate the loss function."""
        criterion = nn.MSELoss()
        return criterion(x_recon, x)

    def get_kl_divergence(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """Calculate the KL divergence."""
        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return kl_divergence / mu.size(0)

    def latent_space(self, x: Tensor) -> Tensor:
        """Get the latent space representation."""
        mean, _ = self.encode(x)
        latent_space = mean.detach().cpu().numpy()
        latent_space = (latent_space - latent_space.mean(axis=0)) / latent_space.std(
            axis=0
        )
        return latent_space
