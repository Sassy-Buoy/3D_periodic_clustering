"""Lightning module for training and evaluation."""

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import lightning as L
from helpers_mod import MCSims
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


class MCSimsDataModule(L.LightningDataModule):
    """Lightning data module for loading and preprocessing data."""

    def __init__(self, batch_size, num_workers):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.data = MCSims()

    def setup(self, stage=None):
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

        # save config.json to hparams.yaml
        self.save_hyperparameters(self.model.config)

    def training_step(self, batch, batch_idx):
        torch.autograd.set_detect_anomaly(True)
        loss = self.model.get_loss(batch, self.model(batch))
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.model(batch))
        self.log("val_loss", loss)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-4)

        scheduler = ExponentialLR(optimizer, gamma=0.98)

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }

    def on_fit_end(self):
        # save the encoded and decoded data
        encoded_data = []
        decoded_data = []

        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(self. trainer.datamodule.test_dataloader()):
                encoded_batch = self.model.encoder(batch)
                decoded_batch = self.model.decoder(encoded_batch)
                encoded_data.append(encoded_batch)
                decoded_data.append(decoded_batch)

        # Concatenate encoded and decoded data
        encoded_data = torch.cat(encoded_data, dim=0)
        decoded_data = torch.cat(decoded_data, dim=0)

        # Save encoded and decoded data
        torch.save(encoded_data, "saved_models/encoded_data.pth")
        print("Encoded data saved")
        torch.save(decoded_data, "saved_models/decoded_data.pth")
        print("Decoded data saved")


class LitVAE(LitAE):
    """Lightning module for training and evaluation of a VAE."""

    def training_step(self, batch, batch_idx):
        batch_recon, mu, log_var = self.model(batch)
        recon_loss = self.model.get_recon_loss(batch, batch_recon)
        kl_loss = self.model.get_kl_divergence(mu, log_var)
        train_loss = recon_loss + kl_loss
        self.log("train_recon_loss", recon_loss)
        self.log("train_kl_loss", kl_loss)
        self.log("train_loss", train_loss)
        return train_loss

    def validation_step(self, batch, batch_idx):
        batch_recon, mu, log_var = self.model(batch)
        recon_loss = self.model.get_recon_loss(batch, batch_recon)
        kl_loss = self.model.get_kl_divergence(mu, log_var)
        val_loss = recon_loss + kl_loss
        self.log("val_recon_loss", recon_loss)
        self.log("val_kl_loss", kl_loss)
        self.log("val_loss", val_loss)
