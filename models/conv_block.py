import torch
from torch import Tensor, nn


class ResBlock(ConvBlock):
    """Residual block with convolutional layers and a skip connection."""

    def __init__(
        self,
        kernels: list[int],
        in_channels: int,
        out_channels: int,
        in_size: int,
        out_size: int,
    ):
        super(ResBlock, self).__init__(
            kernels, in_channels, out_channels, in_size, out_size
        )
        self.skip = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm3d(out_channels),
        )
        self.layers = self.layers[:-1]  # remove the last activation function

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the residual block."""
        if hasattr(self, "upsample"):
            x = self.upsample(x)
        identity = self.skip(x)
        out = self.layers(x)
        out += identity  # Residual connection
        out = nn.GELU()(out)
        if hasattr(self, "downsample"):
            out = self.downsample(out)
        return out


class SpatialAttentionBlock(nn.Module):
    """Spatial attention block to emphasize important spatial regions."""

    def __init__(
        self,
        kernels: list[int],
        in_channels: int,
        out_channels: int,
        in_size: int,
        out_size: int,
    ):
        super(SpatialAttentionBlock, self).__init__()

        kernel_size = 7

        # Ensure kernel_size is odd for proper padding
        assert kernel_size % 2 == 1, "kernel_size must be an odd number."

        self.spatial_attention = nn.Sequential(
            nn.Conv3d(
                in_channels=2,  # Concatenation of max and average pooling outputs
                out_channels=1,
                kernel_size=kernel_size,
                padding="same",
            ),
            nn.Sigmoid(),  # Normalize attention map to [0, 1]
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the spatial attention block.

        Args:
            x (Tensor): Input feature map of shape (B, C, D, H, W).

        Returns:
            Tensor: Output feature map after applying spatial attention.
        """
        # Compute max and average pooling along the channel dimension
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # Shape: (B, 1, D, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # Shape: (B, 1, D, H, W)

        # Concatenate along channel dimension
        concat = torch.cat([max_pool, avg_pool], dim=1)  # Shape: (B, 2, D, H, W)

        # Generate attention map
        attention_map = self.spatial_attention(concat)  # Shape: (B, 1, D, H, W)

        # Apply attention map to input
        return x * attention_map


class CBAM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBAM, self).__init__()
        self.conv_block = ConvBlock(
            kernels=[3, 3, 3],
            in_channels=in_channels,
            out_channels=out_channels,
            in_size=64,
            out_size=64,
        )
        self.spatial_attention = SpatialAttentionBlock(
            kernels=[3, 3, 3],
            in_channels=out_channels,
            out_channels=out_channels,
            in_size=64,
            out_size=64,
        )
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(out_channels, out_channels // 2, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the Convolutional Block Attention Module (CBAM).

        Args:
            x (Tensor): Input feature map of shape (B, C, D, H, W).

        Returns:
            Tensor: Output feature map after applying CBAM.
        """
        x = self.conv_block(x)
        x = self.spatial_attention(x)
        spatial_output = x
        x = torch.mean(x, dim=1, keepdim=True)  # Global average pooling
        channel_output = self.channel_attention(x)
        return spatial_output
