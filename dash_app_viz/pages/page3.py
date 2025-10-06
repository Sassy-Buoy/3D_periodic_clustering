import torch

# ---- Load data ----
latent_space_per_epoch = torch.load("dash_app_viz/latent_space_per_epoch.pth")


# ---- Helper: tensor → Latent space at epoch ----
def plot_latent_space(epoch=999):
    latent_space = latent_space_per_epoch[epoch]
    # Plot the latent space using matplotlib
    import matplotlib.pyplot as plt

    plt.scatter(latent_space[:, 0], latent_space[:, 1])
    plt.show()
