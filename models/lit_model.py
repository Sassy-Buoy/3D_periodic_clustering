"""Lightning module for training and evaluation."""

from torch.optim import Adam
import lightning as L


class LitAE(L.LightningModule):
    """Lightning module for training and evaluation."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.model(batch))
        self.log("train_loss", loss, on_epoch=True, on_step=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.model(batch))
        self.log("val_loss", loss, on_epoch=True, on_step=False, sync_dist=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer


class LitVAE(LitAE):
    """Lightning module for training and evaluation of a VAE."""

    def training_step(self, batch, batch_idx):
        batch_recon, mu, log_var = self.model(batch)
        recon_loss = self.model.get_recon_loss(batch, batch_recon)
        kl = self.model.get_kl_divergence(mu, log_var)
        self.log("train_loss", recon_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("train_kl_loss", kl, on_epoch=True, on_step=False, sync_dist=True)
        return recon_loss + kl

    def validation_step(self, batch, batch_idx):
        batch_recon, mu, log_var = self.model(batch)
        recon_loss = self.model.get_recon_loss(batch, batch_recon)
        kl = self.model.get_kl_divergence(mu, log_var)
        self.log("val_loss", recon_loss, on_epoch=True, on_step=False, sync_dist=True)
        self.log("val_kl_loss", kl, on_epoch=True, on_step=False, sync_dist=True)
