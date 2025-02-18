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

lit_model = LitAE(AutoEncoder(config))

# Define the logger
logger = CSVLogger("")

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="train_loss",  # Metric to monitor
    dirpath=logger.log_dir,  # Directory to save the checkpoints
    filename="model-{epoch:02d}-{val_loss:.2f}",  # Checkpoint filename
    save_top_k=3,  # Save only the best model
    mode="min",  # Minimize the monitored metric
    auto_insert_metric_name=False,  # Avoid automatic metric name insertion in filename
)

trainer = L.Trainer(
    precision="16-mixed",
    callbacks=[checkpoint_callback],
    max_epochs=400,
    accumulate_grad_batches=2,
    logger=logger,
    log_every_n_steps=1,
)

trainer.fit(
    lit_model,
    # ckpt_path="lightning_logs/version_3/autoencoder-99-0.52.ckpt",
    datamodule=MCSimsDataModule(batch_size=64, num_workers=3),
)

# Save the model configuration to the logger directory.
with open(f"{logger.log_dir}/AE_config.json", "w") as f:
    json.dump(config, f)
