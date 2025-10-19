import optuna
from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
import os
from cluster_acc import purity, adj_rand_index

# Load dataset
from models.lit_model import LitModel, MCSimsDataModule

version = "7"
path = f"lightning_logs/version_{version}/"

# get the latest checkpoint from the path folder
checkpoint = max(
    [f for f in os.listdir(path) if f.endswith(".ckpt")],
    key=lambda x: int(x.split("-")[1]),
)
print(f"Loading checkpoint: {checkpoint}")

litmodel = LitModel.load_from_checkpoint(
    os.path.join(path, checkpoint),
)

model = litmodel.model
model.eval()

dataset = MCSimsDataModule(batch_size=256, num_workers=4)
encoded_data = litmodel.encode_data(model, dataset.test_dataloader())

# Objective function for Optuna
def objective(trial):
    # Hyperparameters
    weight_concentration_prior_type = trial.suggest_categorical(
        "weight_concentration_prior_type",
        ["dirichlet_process", "dirichlet_distribution"],
    )
    weight_concentration_prior = trial.suggest_float(
        "weight_concentration_prior", 1e-3, 1e3, log=True
    )
    covariance_type = trial.suggest_categorical(
        "covariance_type", ["full", "tied", "diag", "spherical"]
    )

    labels = (
        BayesianGaussianMixture(
            n_components=5,
            weight_concentration_prior_type=weight_concentration_prior_type,
            weight_concentration_prior=weight_concentration_prior,
            covariance_type=covariance_type,
            max_iter=1000,
        )
        .fit(encoded_data)
        .predict(encoded_data)
    )
    predicted_labels = [labels[10 * i] for i in range(261)]

    return adj_rand_index(predicted_labels)


# Run the optimization
study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
)
study.optimize(objective, n_trials=200)

# Best result
print("Best trial:")
print(study.best_trial)
