"""Autoencoder models for 3D image data."""

import torch
from torch import Tensor, nn

from models.conv_block import ConvBlock, ResBlock


class AutoEncoder(nn.Module):
    """Autoencoder class that inherits from PyTorch's nn.Module class."""

    def __init__(
        self,
        config: dict,
    ):
        super(AutoEncoder, self).__init__()
        self.latent_dim = config["latent_dim"]
        self.layers = config["layers"]

        self.prelatent_dim = (
            self.layers[-1]["out_channels"] * self.layers[-1]["out_size"] ** 3
        )

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()

        for i, layer in enumerate(self.layers):
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
        # Flatten and linear layers for the latent space
        self.encoder.append(nn.Flatten())

        if self.prelatent_dim != self.latent_dim:
            self.encoder.append(nn.Linear(self.prelatent_dim, self.latent_dim))
        self.decoder.insert(
            0,
            nn.Unflatten(
                1,
                (
                    self.layers[-1]["out_channels"],
                    self.layers[-1]["out_size"],
                    self.layers[-1]["out_size"],
                    self.layers[-1]["out_size"],
                ),
            ),
        )
        if self.prelatent_dim != self.latent_dim:
            self.decoder.insert(0, nn.Linear(self.latent_dim, self.prelatent_dim))

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

    def __init__(self, config: dict):
        super(VarAutoEncoder, self).__init__(config)
        self.fc_mean = nn.Linear(self.latent_dim, self.latent_dim)
        self.fc_log_var = nn.Linear(self.latent_dim, self.latent_dim)

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


def encode_data(model, dataloader):
    """Encode the data using the trained VAE or AE."""
    encoded_data = []
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = batch.float()
            encoded_batch = model.encoder(batch)
            encoded_data.append(encoded_batch)
    # Concatenate encoded and decoded data
    encoded_data = torch.cat(encoded_data, dim=0)
    # Save encoded and decoded data
    torch.save(encoded_data, "encoded_data.pth")
    print("Encoded data saved")
