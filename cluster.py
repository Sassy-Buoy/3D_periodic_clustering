import optuna
import torch
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from sklearn.mixture import BayesianGaussianMixture
import pandas as pd

# Load dataset
encoded_data = torch.load("lightning_logs/version_11/encoded_data.pth").cpu().numpy()
true_df = pd.read_csv("dataset.csv")
true_labels = true_df["label"].values


# Define purity score
def purity_score(y_true, y_pred):
    y_pred = [y_pred[10 * i] for i in range(261)]
    matrix = contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(matrix, axis=0)) / np.sum(matrix)

# Objective function for Optuna
def objective(trial):
    # Hyperparameters
    n_components = trial.suggest_int("n_components", 2, 10)
    weight_concentration_prior_type = trial.suggest_categorical("weight_concentration_prior_type", ["dirichlet_process", "dirichlet_distribution"])
    weight_concentration_prior = trial.suggest_float("weight_concentration_prior", 1e-3, 1e3, log=True)
    covariance_type = trial.suggest_categorical("covariance_type", ["full", "tied", "diag", "spherical"])

    labels = BayesianGaussianMixture(
        n_components=n_components,
        weight_concentration_prior_type=weight_concentration_prior_type,
        weight_concentration_prior=weight_concentration_prior,
        covariance_type=covariance_type,
        max_iter=1000,
        random_state=42
    ).fit(encoded_data).predict(encoded_data)

    return purity_score(true_labels, labels)

# Run the optimization
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=200)

# Best result
print("Best trial:")
print(study.best_trial)
