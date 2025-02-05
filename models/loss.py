import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure as ssim

class CustomLoss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        Custom loss function for 3D vector fields.
        SSIM is applied to the magnitude of the vector field.
        Cosine similarity is applied to the direction of the vector field.

        Args:
            alpha (float): Weighting factor between SSIM and cosine similarity losses.
        """
        super(CustomLoss, self).__init__()
        self.alpha = alpha

    def ssim_mag(self, pred, target):
        """
        Computes the SSIM loss for the magnitude of the vector field.

        Args:
            pred (torch.Tensor): Predicted vector field (B x C x H x W x D).
            target (torch.Tensor): Ground truth vector field (B x C x H x W x D).

        Returns:
            torch.Tensor: SSIM loss value.
        """
        pred_mag = torch.norm(pred, dim=1, keepdim=True)
        target_mag = torch.norm(target, dim=1, keepdim=True)
        ssim_value = ssim(pred_mag, target_mag, data_range=2)
        return 1 - ssim_value

    def cosine_similarity_loss(self, pred, target, eps=1e-8):
        """
        Computes cosine similarity loss for 3D vector fields.

        Args:
            pred (torch.Tensor): Predicted vector field (B x C x H x W x D).
            target (torch.Tensor): Ground truth vector field (B x C x H x W x D).
            eps (float): Small value to avoid division by zero.

        Returns:
            torch.Tensor: Cosine similarity loss value.
        """
        pred_norm = F.normalize(pred, dim=1, eps=eps)
        target_norm = F.normalize(target, dim=1, eps=eps)
        cos_sim = torch.sum(pred_norm * target_norm, dim=1)  # Cosine similarity per voxel
        loss = (1 - cos_sim).mean()  # Average over all voxels and batch
        return loss

    def forward(self, pred, target):
        """
        Computes the custom loss function.

        Args:
            pred (torch.Tensor): Predicted vector field.
            target (torch.Tensor): Ground truth vector field.

        Returns:
            torch.Tensor: Custom loss value.
        """
        loss_ssim = self.ssim_mag(pred, target)
        loss_cosine = self.cosine_similarity_loss(pred, target)
        total_loss = self.alpha * loss_ssim + (1 - self.alpha) * loss_cosine
        return total_loss
