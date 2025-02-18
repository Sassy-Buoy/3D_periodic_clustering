"""Lightning module for training and evaluation."""

import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader

import lightning as L
from helpers_mod import MCSims
from models.auto_encoder import encode_data


class MCSimsDataModule(L.LightningDataModule):
    """Lightning data module for loading and preprocessing data."""

    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.data = MCSims()

    def setup(self, stage=None):
        if not hasattr(self, "data"):
            self.prepare_data()
        self.train_data, self.val_data = train_test_split(
            self.data, test_size=0.2, random_state=42
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class LitAE(L.LightningModule):
    """Lightning module for training and evaluation."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        batch = batch.float()
        torch.autograd.set_detect_anomaly(True)
        loss = self.model.get_loss(batch, self.model(batch))
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch.float()
        loss = self.model.get_loss(batch, self.model(batch))
        self.log("val_loss", loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        # scheduler = ExponentialLR(optimizer, gamma=0.99)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return optimizer

    def on_fit_end(self):
        encode_data(self.model, self.trainer.datamodule.test_dataloader())
        torch.save(self.model.state_dict(), "model.pth")


class LitVAE(LitAE):
    """Lightning module for training and evaluation of a VAE."""

    def training_step(self, batch, batch_idx):
        batch = batch.float()
        batch_recon, mu, log_var = self.model(batch)
        recon_loss = self.model.get_recon_loss(batch, batch_recon)
        kl_loss = self.model.get_kl_divergence(mu, log_var)
        train_loss = recon_loss + kl_loss
        self.log("train_recon_loss", recon_loss, sync_dist=True)
        self.log("train_kl_loss", kl_loss, sync_dist=True)
        self.log("train_loss", train_loss, sync_dist=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        batch = batch.float()
        batch_recon, mu, log_var = self.model(batch)
        recon_loss = self.model.get_recon_loss(batch, batch_recon)
        kl_loss = self.model.get_kl_divergence(mu, log_var)
        val_loss = recon_loss + kl_loss
        self.log("val_recon_loss", recon_loss, sync_dist=True)
        self.log("val_kl_loss", kl_loss, sync_dist=True)
        self.log("val_loss", val_loss, sync_dist=True)
