"""Autoencoder models for 3D image data."""

from math import ceil

import torch
from torch import Tensor, nn


class AutoEncoder(nn.Module):
    """Autoencoder class that inherits from PyTorch's nn.Module class."""

    def __init__(self, layers: list[dict]):
        super(AutoEncoder, self).__init__()

        self.encoder = []
        self.decoder = []

        for i, layer in enumerate(layers):
            # Insert the encoder layers
            if layer["type"] == "Conv":
                self.encoder += self._conv_block(
                    in_channels=layer["in_channels"],
                    out_channels=layer["out_channels"],
                    kernel_size=layer["kernel_size"],
                    stride=layer["stride"],
                )
                # Insert the decoder layers in reverse order
                self.decoder = (
                    self._conv_block(
                        in_channels=layer["out_channels"],
                        out_channels=layer["in_channels"],
                        kernel_size=layer["kernel_size"],
                        stride=layer["stride"],
                        transpose=True,
                    )
                    + self.decoder
                )
            elif layer["type"] == "Flatten":
                self.encoder.append(nn.Flatten())
                self.decoder.insert(
                    0,
                    nn.Unflatten(
                        1,
                        (
                            layer["in_channels"],
                            layer["in_size"],
                            layer["in_size"],
                            layer["in_size"],
                        ),
                    ),
                )
            elif layer["type"] == "Linear":
                self.encoder += self._linear_block(
                    in_features=layer["in_features"],
                    out_features=layer["out_features"],
                    activation=layer["activation"],
                )
                self.decoder = (
                    self._linear_block(
                        in_features=layer["out_features"],
                        out_features=layer["in_features"],
                        activation=layer["activation"],
                    )[::-1]
                    + self.decoder
                )

        self.decoder.pop(-1)
        self.decoder.append(nn.Tanh())

        # convert the lists to nn.Sequential
        self.encoder = nn.Sequential(*self.encoder)
        self.decoder = nn.Sequential(*self.decoder)

    def _conv_block(
        self, in_channels, out_channels, kernel_size, stride, transpose=False
    ):
        """Create a convolutional block."""
        return [
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=ceil((kernel_size - stride) / 2),
            )
            if not transpose
            else nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=ceil((kernel_size - stride) / 2),
                output_padding=0 if ((kernel_size - stride) % 2) == 0 else 1,
            ),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        ]

    def _linear_block(self, in_features, out_features, activation):
        """Create a linear block."""
        layers = [nn.Linear(in_features, out_features)]
        if activation == "ReLU":
            layers.append(nn.ReLU())
        elif activation == "LeakyReLU":
            layers.append(nn.LeakyReLU())
        return layers

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

    def __init__(self, layers: list[dict]):
        super(VarAutoEncoder, self).__init__(layers)
        self.latent_dim = layers[-1]["out_features"]
        self.fc_mean = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(self.latent_dim, self.latent_dim)
        nn.init.xavier_uniform_(self.fc_mean.weight)
        nn.init.xavier_uniform_(self.fc_log_var.weight)
        nn.init.constant_(self.fc_mean.bias, 0)
        nn.init.constant_(self.fc_log_var.bias, -5)

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


class GMVAE(VarAutoEncoder):
    """Gaussian Mixture Variational Autoencoder class."""

    def __init__(self, layers: list[dict]):
        super(GMVAE, self).__init__(layers)

        self.gmm_means = nn.Parameter(torch.randn(self.latent_dim, self.latent_dim))
        self.gmm_log_vars = nn.Parameter(torch.zeros(self.latent_dim, self.latent_dim))
        self.gmm_weights = nn.Parameter(torch.ones(self.latent_dim) / self.latent_dim)
    
    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        pass


def encode_data(model, dataloader):
    """Encode the data using the trained VAE or AE."""
    encoded_data = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.float()
            encoded_batch = model.encoder(batch.to("cuda"))
            encoded_data.append(encoded_batch)
    # Concatenate encoded and decoded data
    encoded_data = torch.cat(encoded_data, dim=0)
    # Save encoded and decoded data
    return encoded_data
