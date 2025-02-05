"""run.py"""

import json

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from models.auto_encoder import AutoEncoder
from models.lit_model import LitAE, MCSimsDataModule


# Load the configuration
with open("config.json", "r") as f:
    config = json.load(f)

batch_size = config["hyper_params"]["batch_size"]
# lr = config["hyper_params"]["lr"]

autoencoder = AutoEncoder(config["model_params"])

lit_model = LitAE(autoencoder)
# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Metric to monitor
    dirpath="checkpoints/",  # Directory to save the model
    filename="best-checkpoint",  # Filename for the best model
    save_top_k=1,  # Save only the best model
    mode="min",  # Minimize the monitored metric (val_loss)
    auto_insert_metric_name=False,  # Avoid automatic metric name insertion in filename
)

trainer = L.Trainer(
    precision=64,
    callbacks=[EarlyStopping(monitor="val_loss", patience=5), checkpoint_callback],
    accumulate_grad_batches=2,
    num_sanity_val_steps=0,
    logger=CSVLogger("logs"),
    log_every_n_steps=1,
)

trainer.fit(
    lit_model,
    # ckpt_path="checkpoints/best-checkpoint-vn.ckpt",
    datamodule=MCSimsDataModule(batch_size=batch_size, num_workers=3),
)
