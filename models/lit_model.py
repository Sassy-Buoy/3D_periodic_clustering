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
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model.get_loss(batch, self.model(batch))
        self.log("val_loss", loss, sync_dist=True)

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=1e-3)
        return optimizer
