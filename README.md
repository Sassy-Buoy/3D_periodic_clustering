# ML-based Clustering and Dimensionality Reduction for 3D Magnetic Field Simulations

[![Render](https://img.shields.io/badge/Live%20App-Render-blue?style=for-the-badge&logo=render)](https://threed-periodic-clustering.onrender.com)

This repository contains code for clustering and dimensionality reduction of 3D magnetic field simulation data using machine learning techniques. The main components include:
- **Data Handling**: Functions to load and preprocess 3D magnetic field data.
- **Model Definitions**: Implementation of Variational Autoencoders (VAE) and Autoencoders (AE) for dimensionality reduction.
- **Training Pipeline**: Scripts to train the models using PyTorch Lightning.
- **Clustering**: Methods to perform clustering on the latent space representations.
- **Hyperparameter Tuning**: Tools to optimize model hyperparameters.
- **Visualization**: Tools to visualize the clustering results and latent space representations.

## Micromagnetic Simulations


## Project Structure

### Data Directory (**`data/`**)

- **`dataset.py`**
- **`dataset.csv`**
- **`center_plots`**

### Models Directory (**`models/`**)

- **`__init__.py`**: Package initialization, exports main classes
- **`auto_encoder.py`**: Implementation of autoencoder and variational autoencoder architectures
- **`lit_model.py`**: PyTorch Lightning module for training, validation, and data handling

## Plotting Directory (**`plot/`**)

- **`__init__.py`**: Package initialization, exports main classes
- **`training_log.py`**: Functions to visualize training logs and metrics
- **`plot.py`**: Functions to visualize clustering results and latent space representations

### Training Logs (**`lightning_logs/`**)

- **`version_*/`**: PyTorch Lightning training logs, checkpoints, and metrics for different training runs. Model checkpoints are automatically saved based on validation performance.

### Root Files

- **`run.py`**: Main training script that loads configuration from `config.yaml` and trains the autoencoder model using PyTorch Lightning
- **`config.yaml`**: Configuration file defining model architecture, hyperparameters, and training settings
- **`cluster_acc.py`**: Clustering evaluation metrics including purity score and adjusted Rand index
- **`run.sh`**: Shell script for running training jobs (likely for HPC environments)
- **`test.ipynb`**: Jupyter notebook for testing and experimentation
