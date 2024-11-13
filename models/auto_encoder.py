"""Vanilla Encoder class."""

from torch import nn, Tensor


class AutoEncoder(nn.Module):
    """Autoencoder class that inherits from PyTorch's nn.Module class."""

    def __init__(
        self,
        kernels: list[list[int]],
        batch_norm: bool = True,
        activation: str = "Relu",
        latent_dim: int = 12,
    ):
        super(AutoEncoder, self).__init__()
        self.kernels = kernels
        self.batch_norm = batch_norm
        if activation == "Relu":
            self.activation = nn.ReLU()
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        for i, kernel_sizes in enumerate(self.kernels):
            # Insert the encoder layers
            self.encoder.append(
                self._conv_block(
                    kernel_sizes, 3 if i == 0 else 2 ** (i + 1), 2 ** (i + 2)
                )
            )
            self.encoder.append(nn.MaxPool3d(kernel_size=2, stride=2))
            # Insert the decoder layers in reverse order
            self.decoder.insert(
                0,
                self._conv_block(
                    kernel_sizes, 2 ** (i + 2), 3 if i == 0 else 2 ** (i + 1)
                ),
            )
            if i == 0:
                self.decoder.insert(0, nn.Upsample(size=(97, 97, 97)))
            elif i == len(self.kernels) - 1:
                self.decoder.insert(0, nn.Upsample(scale_factor=3))
            else:
                self.decoder.insert(0, nn.Upsample(scale_factor=2))
        # Flatten and unflatten the tensors
        self.encoder.append(nn.Flatten())
        self.decoder.insert(0, nn.Unflatten(1, (2 ** (len(self.kernels) + 1), 1, 1, 1)))

    def _conv_block(
        self, kernel_sizes: list, in_channels: int, out_channels: int
    ) -> nn.Sequential:
        """Helper function to create a convolutional block."""
        layers = nn.Sequential()
        for layer, kernel_size in enumerate(kernel_sizes):
            layers.append(
                nn.Conv3d(
                    in_channels=in_channels if layer == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding="same",
                )
            )
            if self.batch_norm:
                layers.append(nn.BatchNorm3d(out_channels))
            layers.append(self.activation)
        return layers

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the autoencoder."""
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def get_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Calculate the loss function."""
        criterion = nn.MSELoss()
        loss = criterion(x_recon, x)
        return loss
