import os
import pathlib as pl

import discretisedfield as df
import holoviews as hv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import mplcursors
import pandas as pd
import plotly.graph_objects as go

from dataset import MCSims


def plot_field_xy_from_tensor(tensor, save_path: str | None = None):
    """
    Convert a ``torch.Tensor`` to ``discretisedfield.Field`` and plot the xy-plane cut.
    """
    x_dim, y_dim, z_dim, v_dim = (
        tensor.shape[1],
        tensor.shape[2],
        tensor.shape[3],
        tensor.shape[0],
    )
    value = tensor.to("cpu").numpy().transpose((1, 2, 3, 0))
    p1, p2 = (0, 0, 0), (x_dim - 1, y_dim - 1, z_dim - 1)
    mesh = df.Mesh(p1=p1, p2=p2, n=(x_dim, y_dim, z_dim))
    plot = df.Field(mesh=mesh, value=value, nvdim=v_dim).hv(
        kdims=["x", "y"], scalar_kw={"clim": (-1, 1)}
    )
    if save_path:
        hv.save(plot, save_path)
    return plot


def plot_H_vs_T_with_hover(labels: list = []):
    """
    Plots H vs T from the dataset dataframe using Matplotlib.
    Displays preloaded images on the right when hovering over points.
    """
    dataset = MCSims()  # Ensure this class or object is defined and has the data frame
    df = dataset.data_frame
    df["cluster"] = labels

    # Create a list of unique clusters
    unique_clusters = df["cluster"].unique()

    # Assign a color for each cluster (using a colormap, e.g., 'viridis')
    colors = [plt.cm.jet(i / len(unique_clusters)) for i in range(len(unique_clusters))]

    # Map the cluster labels to the corresponding colors
    color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
    df["color"] = df["cluster"].map(color_map)

    # Preload images into a dictionary
    image_cache = {}
    for index in df.index:
        img_path = f"center_plots/{index}.png"
        if os.path.exists(img_path):
            image_cache[index] = mpimg.imread(img_path)

    # Create figure with two subplots: scatter plot (left) & image display (right)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    ax_scatter, ax_image = axes

    scatter = ax_scatter.scatter(
        df["T (K)"],
        df["H (A/m)"],
        c=df["color"],
        marker="o",
    )

    ax_scatter.set_xlabel("H (A/m)")
    ax_scatter.set_ylabel("T (K)")
    ax_scatter.set_title("H vs T Phase Diagram")

    # Configure image display panel
    ax_image.axis("off")  # Hide axes
    img_display = ax_image.imshow([[0]])  # Placeholder image

    cursor = mplcursors.cursor(scatter, hover=True)

    @cursor.connect("add")
    def on_hover(sel):
        """Updates the right-side image when hovering over a point."""
        index = sel.index
        if index in image_cache:
            img_display.set_data(image_cache[index])
            ax_image.set_visible(True)
        else:
            ax_image.set_visible(False)

        fig.canvas.draw_idle()

    plt.show()


def training_log(version: list = [], y_range: tuple | None = None):
    """
    Plots training and validation loss from a Lightning log directory.

    - Automatically imports all metrics.csw files for different versions in the lightning_logs/version_*/ directories.
    - Plots training and validation loss for each version.
    - Checkboxes to toggle visibility of each version.
    - Checkboxes to toggle visibility of train_loss, val_loss
        and also train_recon_loss, train_kl_loss, val_recon_loss, val_kl_loss if available.
    - x - axis is the epoch number. Calculate the number of steps for each epoch automatically for each version.
    """
    # Load all CSV files in the lightning_logs directory

    data = []
    if len(version) == 0:
        for i, path in enumerate(pl.Path("lightning_logs").glob("version_*")):
            try:
                data.append(pd.read_csv(path / "metrics.csv"))
            except FileNotFoundError:
                print(f"No metrics.csv file found in {path}")
    else:
        for i, v in enumerate(version):
            path = pl.Path(f"lightning_logs/version_{v}")
            try:
                data.append(pd.read_csv(path / "metrics.csv"))
            except FileNotFoundError:
                print(f"No metrics.csv file found in {path}")

    # Create the figure
    fig = go.Figure()

    for i, df in enumerate(data):
        # aggregate over steps
        df = df.groupby("epoch").mean().reset_index()

        # Add training loss
        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["train_loss"],
                mode="lines",
                name=f"train_loss_{i}",
                visible="legendonly",
            )
        )

        # Add validation loss
        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["val_loss"],
                mode="lines",
                name=f"val_loss_{i}",
                # visible="legendonly",
            )
        )

        # Add training reconstruction loss if available
        if "train_recon_loss" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["epoch"],
                    y=df["train_recon_loss"],
                    mode="lines",
                    name=f"train_recon_loss_{i}",
                    visible="legendonly",
                )
            )

        # Add training KL loss if available
        if "train_kl_loss" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["epoch"],
                    y=df["train_kl_loss"],
                    mode="lines",
                    name=f"train_kl_loss_{i}",
                    visible="legendonly",
                )
            )

        # Add validation reconstruction loss if available
        if "val_recon_loss" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["epoch"],
                    y=df["val_recon_loss"],
                    mode="lines",
                    name=f"val_recon_loss_{i}",
                    visible="legendonly",
                )
            )

        # Add validation KL loss if available
        if "val_kl_loss" in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["epoch"],
                    y=df["val_kl_loss"],
                    mode="lines",
                    name=f"val_kl_loss_{i}",
                    visible="legendonly",
                )
            )

    # Update layout
    fig.update_layout(
        title="Training and Validation Loss",
        xaxis_title="Epoch",
        yaxis_title="Loss",
        yaxis_range=y_range,
        
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "All",
                        "method": "update",
                        "args": [{"visible": [True] * len(fig.data)}],
                    },
                    {
                        "label": "None",
                        "method": "update",
                        "args": [{"visible": [False] * len(fig.data)}],
                    },
                ],
                "direction": "down",
                "showactive": True,
            },
        ],
    )

    # Show the figure
    fig.show()


def save_center_slices(dataset: MCSims, save_folder: str = "center_plots/"):
    """
    Extract the center slice along the Z-axis from each tensor,
    plot it without axes or white space, and save it in the specified folder.
    """
    os.makedirs(save_folder, exist_ok=True)

    hv.extension("matplotlib")

    for index in range(len(dataset)):
        tensor = dataset[index]  # Get tensor

        # Save figure without padding or white space
        file_path = os.path.join(save_folder, f"{index}.png")
        plot_center_slice(tensor, save_path=file_path)
        print(f"Saved: {file_path}")


def plot_center_slice(tensor, save_path: str | None = None):
    """
    Convert a ``torch.Tensor`` to ``discretisedfield.Field`` and plot the center z-slice.
    - Uses z-component as a scalar field (color map).
    - Uses (x, y) components as a vector field (arrows).
    """
    x_dim, y_dim, z_dim, v_dim = (
        tensor.shape[1],
        tensor.shape[2],
        tensor.shape[3],
        tensor.shape[0],
    )

    center_z = z_dim // 2  # Find the center z-index
    value = tensor.to("cpu").numpy().transpose((1, 2, 3, 0))[:, :, center_z, :]

    p1, p2 = (0, 0), (x_dim - 1, y_dim - 1)
    mesh = df.Mesh(p1=p1, p2=p2, n=(x_dim, y_dim))

    # Extract x, y components for vector field & z component for scalar field
    vector_field = value[..., :2]  # First two components (x, y)
    scalar_field = value[..., 2]  # Third component (z)

    # Create fields
    vector_df = df.Field(mesh=mesh, value=vector_field, nvdim=2)
    scalar_df = df.Field(mesh=mesh, value=scalar_field, nvdim=1)

    # Plot both fields together
    vector_plot = vector_df.hv(kdims=["x", "y"], vector_kw={"scale": 1})
    scalar_plot = scalar_df.hv(
        kdims=["x", "y"], scalar_kw={"clim": (-1, 1), "colorbar": False}
    )

    plot = scalar_plot * vector_plot  # Overlay plots
    plot.opts(
        hv.opts.Image(xaxis=None, yaxis=None, axiswise=True),
        hv.opts.VectorField(xaxis=None, yaxis=None, axiswise=True),
    )

    fig = hv.render(plot)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save as PNG without requiring a browser
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)

    return plot


if __name__ == "__main__":
    dataset = MCSims(preload=False, preprocess=False, augment=False)
    save_center_slices(dataset)
