"""Vanilla Encoder class."""

from torch import nn, Tensor, flatten, reshape


class Autoencoder(nn.Module):
    """Autoencoder class that inherits from PyTorch's nn.Module class."""

    def __init__(self, kernels: list[list[int]],
                 batch_norm: bool = True,
                 activation: str = 'Relu'):
        super(Autoencoder, self).__init__()
        self.kernels = kernels
        self.batch_norm = batch_norm
        if activation == 'Relu':
            self.activation = nn.ReLU()
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.bottle_neck = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_en = nn.Linear(96, 12)
        self.fc_de = nn.Linear(12, 96)

    def _conv_block(self, x: Tensor, kernel_sizes: list, resnet: bool = False) -> Tensor:
        """Helper function to create a convolutional block."""
        if resnet:
            shortcut = x
        for layer, kernel_size in enumerate(kernel_sizes):
            x = nn.Conv3d(x.shape[1], 2*x.shape[1] if layer == 0 else x.shape[1],
                          kernel_size=kernel_size, padding='same')(x)
            if self.batch_norm:
                x = nn.BatchNorm3d(x.shape[1])(x)
            x = self.activation(x)
        if resnet:
            x += shortcut
        return x

    def _deconv_block(self, x: Tensor, kernel_sizes: list, resnet: bool = False) -> Tensor:
        """Helper function to create a deconvolutional block."""
        if resnet:
            shortcut = x
        for layer, kernel_size in enumerate(kernel_sizes):
            x = nn.ConvTranspose3d(x.shape[1], 2*x.shape[1] if layer == 0 else x.shape[1],
                                   kernel_size=kernel_size, padding='same')(x)
            if self.batch_norm:
                x = nn.BatchNorm3d(x.shape[1])(x)
            x = self.activation(x)
        if resnet:
            x += shortcut
        return x

    def encoder(self, x: Tensor) -> Tensor:
        """Forward pass through the encoder."""
        for kernels in self.kernels:
            x = self._conv_block(x, kernels)  # Output: (96, 3, 3, 3)
        x = self.pool(x)  # Output: (96, 1, 1, 1)
        x = self.bottle_neck(x)  # Output: (96, 1, 1, 1)
        x = flatten(x, 1)  # Output: (96)
        x = self.fc_en(x)  # Output: (12)
        return x

    def decoder(self, x: Tensor) -> Tensor:
        """Forward pass through the decoder."""
        x = self.fc_de(x)  # Output: (96)
        x = reshape(x, (-1, 96, 1, 1, 1))
        for kernels in reversed(self.kernels):
            x = self._deconv_block(x, kernels)
        return x

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
