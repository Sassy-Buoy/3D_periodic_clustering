"""Vanilla Encoder class."""

from torch import nn, Tensor, flatten, reshape


class AutoEncoder(nn.Module):
    """Autoencoder class that inherits from PyTorch's nn.Module class."""

    def __init__(self, kernels: list[list[int]],
                 batch_norm: bool = True,
                 activation: str = 'Relu',
                 latent_dim: int = 12):
        super(AutoEncoder, self).__init__()
        self.kernels = kernels
        self.batch_norm = batch_norm
        if activation == 'Relu':
            self.activation = nn.ReLU()
        self.latent_dim = latent_dim

        channels = 3
        self.encoder = nn.Sequential()
        for kernel_sizes in self.kernels:
            channels *= 2
            self.encoder.append(self._conv_block(kernel_sizes, channels))
            self.encoder.append(nn.MaxPool3d(kernel_size=2, stride=2))
        self.encoder.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.encoder.append(nn.Flatten())
        self.encoder.append(nn.Linear(channels, self.latent_dim))

        self.decoder = nn.Sequential()
        self.decoder.append(nn.Linear(self.latent_dim, channels))
        self.decoder.append(nn.Unflatten(1, (channels, 1, 1, 1)))
        self.decoder.append(nn.Upsample(
            scale_factor=3, mode='trilinear', align_corners=True))

        for kernel_sizes in reversed(self.kernels):
            channels //= 2
            self.decoder.append(self._conv_trans_block(kernel_sizes, channels))
            if channels != 3:
                self.decoder.append(nn.Upsample(
                    scale_factor=2, mode='trilinear', align_corners=True))
            else:
                self.decoder.append(nn.Upsample(
                    size=(97, 97, 97), mode='trilinear', align_corners=True))

    def _conv_block(self, kernel_sizes: list, out_channels: int) -> nn.Sequential:
        """Helper function to create a convolutional block."""
        layers = nn.Sequential()
        for layer, kernel_size in enumerate(kernel_sizes):
            layers.append(nn.Conv3d(
                in_channels=out_channels//2 if layer == 0 else out_channels,
                out_channels=out_channels, kernel_size=kernel_size, padding='same'))
            if self.batch_norm:
                layers.append(nn.BatchNorm3d(out_channels))
            layers.append(self.activation)
        return layers

    def _conv_trans_block(self, kernel_sizes: list, out_channels: int) -> nn.Sequential:
        """Helper function to create a deconvolutional block."""
        layers = nn.Sequential()
        for layer, kernel_size in enumerate(kernel_sizes):
            layers.append(nn.ConvTranspose3d(
                in_channels=2*out_channels if layer == 0 else out_channels,
                out_channels=out_channels, kernel_size=kernel_size,
                padding=(kernel_size-1)//2, output_padding=(kernel_size-1) % 2))
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
