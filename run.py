"""run.py"""

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import CSVLogger

from models.lit_model import LitModel, MCSimsDataModule

# get model_type and config from config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)
model_type = config["model_type"]
config = config["config"]

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
    accumulate_grad_batches=2,
    logger=logger,
    log_every_n_steps=1,
)

trainer.fit(
    LitModel(model_type="vanilla", config=config, plot=False),
    # ckpt_path="lightning_logs/version_3/autoencoder-99-0.52.ckpt",
    datamodule=MCSimsDataModule(batch_size=64, num_workers=3),
)
