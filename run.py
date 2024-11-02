""" Train the model and save it """
import torch
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import lightning as L

from helpers import MCSims
from models.auto_encoder import Autoencoder
from models.lit_model import LitAE

dataset = MCSims()

train_set, val_set = dataset.train_test_split()
train_loader = DataLoader(train_set, batch_size=64,
                          shuffle=True, num_workers=3)
val_loader = DataLoader(val_set, batch_size=64,
                        shuffle=False, num_workers=3)

kernels = [[7], [7], [5], [5], [3], [3]]

model = Autoencoder(kernels)

lit_model = LitAE(model)
trainer = L.Trainer(callbacks=[EarlyStopping(monitor="val_loss", patience=5)],
                    log_every_n_steps=1)
trainer.fit(lit_model, train_loader, val_loader)

# save the model
torch.save(lit_model.model.state_dict(), "model.pth")
torch.save(lit_model.model.encoder.state_dict(), "encoder.pt")
torch.save(lit_model.model.decoder.state_dict(), "decoder.pt")
