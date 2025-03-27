"""Lightning module for training and evaluation."""

import os
from math import cos, pi

import imageio
import lightning as L
import matplotlib.pyplot as plt
import torch
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.optim import Adam

# from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset

from dataset import MCSims
from models.auto_encoder import AutoEncoder, VarAutoEncoder, encode_data


class MCSimsDataModule(L.LightningDataModule):
    """Lightning data module for loading and preprocessing data."""

    def __init__(self, batch_size: int, num_workers: int):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.data = MCSims()

    def setup(self, stage=None):
        if not hasattr(self, "data"):
            self.prepare_data()

        if len(self.data) == 2601:
            self.train_data, self.val_data = train_test_split(
                self.data, test_size=0.2, random_state=42
            )
        elif len(self.data) == 2601 * 4:
            original_indices = list(
                range(len(self.data) // 4)
            )  # base indices before augmentation
            train_indices, val_indices = train_test_split(
                original_indices, test_size=0.2, random_state=42
            )

            # Expand the indices to include all rotations
            self.train_data = Subset(
                self.data, [i * 4 + r for i in train_indices for r in range(4)]
            )
            self.val_data = Subset(
                self.data, [i * 4 + r for i in val_indices for r in range(4)]
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
            MCSims(augment=False),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


class LitModel(L.LightningModule):
    """Lightning module for training and evaluation."""

    def __init__(
        self,
        model_type: str,
        config: list,
        beta: float,
        learning_rate: float,
        plot: bool = False,
    ):
        super().__init__()
        self.model_type = model_type
        if model_type == "vanilla":
            self.model = AutoEncoder(config)
        elif model_type == "variational":
            self.model = VarAutoEncoder(config)
        self.plot = plot
        self.learning_rate = learning_rate
        self.beta = beta

        # write the model type and configuration to the hparams.yaml file
        self.save_hyperparameters("model_type", "config", "learning_rate")
        if model_type == "variational":
            self.save_hyperparameters("beta")

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        # scheduler = ExponentialLR(optimizer, gamma=0.99)
        # return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        return optimizer

    def beta_scheduler(self, current_epoch, max_epochs, num_cycles=10):
        """cycle beta from 0 to 1 and back sinusoidally"""
        cycle_length = max_epochs // num_cycles
        cycle_position = current_epoch % cycle_length
        if cycle_position < cycle_length / 2:
            r = (1.0 - cos(pi * cycle_position / cycle_length))
        else:
            r = 1.0
        return r * self.beta

    def training_step(self, batch, batch_idx):
        batch = batch.float()
        torch.autograd.set_detect_anomaly(True)

        if self.model_type == "variational":
            batch_recon, mean, log_var = self.model(batch)
            recon_loss = self.model.get_recon_loss(batch, batch_recon)
            self.log("train_recon_loss", recon_loss, sync_dist=True)

            kl_loss = self.model.get_kl_divergence(mean, log_var)
            self.log("train_kl_loss", kl_loss, sync_dist=True)

            beta = self.beta_scheduler(self.current_epoch, self.trainer.max_epochs)
            loss = recon_loss + beta * kl_loss

        else:
            loss = self.model.get_loss(batch, self.model(batch))

        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        batch = batch.float()

        if self.model_type == "variational":
            batch_recon, mean, log_var = self.model(batch)
            recon_loss = self.model.get_recon_loss(batch, batch_recon)
            self.log("val_recon_loss", recon_loss, sync_dist=True)

            kl_loss = self.model.get_kl_divergence(mean, log_var)
            self.log("val_kl_loss", kl_loss, sync_dist=True)

            beta = self.beta_scheduler(self.current_epoch, self.trainer.max_epochs)
            loss = recon_loss + beta * kl_loss

        else:
            loss = self.model.get_loss(batch, self.model(batch))

        self.log("val_loss", loss, sync_dist=True)

    def on_validation_epoch_end(self):
        if self.plot:
            # plot the latent space
            latent_space = encode_data(
                self.model, self.trainer.datamodule.test_dataloader()
            )
            torch.save(latent_space, f"{self.logger.log_dir}/latent_space.pth")
            # reduce the dimensionality of the latent space from 12 to 2
            tsne = TSNE(n_components=2, random_state=42)
            latent_space = tsne.fit_transform(latent_space.cpu())

            plt.scatter(latent_space[:, 0], latent_space[:, 1])
            plt.title(f"Epoch: {self.current_epoch}")
            if not os.path.exists(f"{self.logger.log_dir}/frames"):
                os.makedirs(f"{self.logger.log_dir}/frames")
            plt.savefig(f"{self.logger.log_dir}/frames/{self.current_epoch}.png")
            plt.close()
        else:
            pass

    def on_fit_end(self):
        # save the model
        torch.save(self.model.state_dict(), f"{self.logger.log_dir}/model.pth")
        print("Model saved")
        encoded_data = []
        self.model.eval()
        with torch.no_grad():
            for i, batch in enumerate(self.trainer.datamodule.test_dataloader()):
                batch = batch.float()
                encoded_batch = self.model.encoder(batch)
                encoded_data.append(encoded_batch)
        # Concatenate encoded and decoded data
        encoded_data = torch.cat(encoded_data, dim=0)
        # Save encoded and decoded data
        torch.save(encoded_data, f"{self.logger.log_dir}/encoded_data.pth")
        if self.plot:
            # turn the images into a gif
            images = []
            for i in range(self.trainer.current_epoch + 1):
                images.append(imageio.imread(f"{self.logger.log_dir}/frames/{i}.png"))
            imageio.mimsave(f"{self.logger.log_dir}/latent_space.gif", images)
