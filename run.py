"""run.py"""

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from models.lit_model import LitModel, MCSimsDataModule

# Define the logger
logger = CSVLogger("")

# Define the checkpoint callback
checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",  # Metric to monitor
    dirpath=logger.log_dir,  # Directory to save the checkpoints
    filename="model-{epoch:02d}-{val_loss:.3f}",  # Checkpoint filename
    save_top_k=3,  # Save only the best model
    mode="min",  # Minimize the monitored metric
    auto_insert_metric_name=False,  # Avoid automatic metric name insertion in filename
)

trainer = L.Trainer(
    precision="16-mixed",
    callbacks=[checkpoint_callback, EarlyStopping(monitor="val_loss", patience=50)],
    min_epochs=1000,
    logger=logger,
    accumulate_grad_batches=2,
    log_every_n_steps=1,
)

# get model_type and config from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

model = LitModel(
    model_type=config["model_type"],
    config=config["config"],
    beta=config["beta"],
    learning_rate=config["learning_rate"],
    plot=False,
)

weights = LitModel.load_from_checkpoint("lightning_logs/version_1/model-510-0.012.ckpt", hparams_file="lightning_logs/version_1/hparams.yaml").state_dict()
model.model.load_state_dict(weights, strict=False)

trainer.fit(
    model,
    datamodule=MCSimsDataModule(batch_size=int(config["batch_size"] / 2), num_workers=4),
)
