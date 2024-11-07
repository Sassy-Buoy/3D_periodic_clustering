""" Train the model and save it """
import torch
from torch.utils.data import DataLoader

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from helpers import MCSims
from models.var_auto_encoder import VarAutoEncoder
from models.lit_model import LitAE

dataset = MCSims()

train_set, val_set = dataset.train_test_split()
train_loader = DataLoader(train_set, batch_size=64,
                          shuffle=True, num_workers=3)
val_loader = DataLoader(val_set, batch_size=64,
                        shuffle=False, num_workers=3)

kernels = [[7], [7], [5], [3], [3]]

VAE = VarAutoEncoder(kernels)
lit_model = LitAE(VAE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lit_model = lit_model.to(device)

trainer = L.Trainer(callbacks=[EarlyStopping(monitor="val_loss")],
                    accelerator="gpu", log_every_n_steps=1)
trainer.fit(lit_model, train_loader, val_loader)

# save the model
torch.save(lit_model.model.state_dict(), "var_model.pth")
