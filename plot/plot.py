import os

import discretisedfield as df
import holoviews as hv
import matplotlib.pyplot as plt

# from sklearn.manifold import TSNE
# import umap
import plotly.graph_objects as go

from data.dataset import MCSims


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


def plot_H_vs_T(labels: list = []):
    """
    Plots H vs T from the dataset dataframe using Matplotlib.
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

    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["T (K)"],
        df["H (A/m)"],
        c=df["color"],
        marker="o",
    )

    plt.xlabel("T (K)")
    plt.ylabel("H (A/m)")
    plt.title("H vs T Phase Diagram")
    plt.show()


def plot_H_vs_T_with_hover(labels: list = []):
    """
    Plots H vs T using Plotly FigureWidget.
    Displays preloaded images on the right when hovering over points.
    """
    # Load dataset
    dataset = MCSims()  # Make sure this class exists and has a `data_frame`
    df = dataset.data_frame.copy()
    df["cluster"] = labels

    # Assign colors to clusters
    import plotly.express as px

    unique_clusters = df["cluster"].unique()
    colors = px.colors.qualitative.Dark24
    color_map = {
        cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)
    }
    df["color"] = df["cluster"].map(color_map)

    # Preload images into a dictionary
    image_cache = {}
    for idx in df.index:
        img_path = f"data/center_plots/{idx}.png"
        if os.path.exists(img_path):
            image_cache[idx] = img_path

    # Create FigureWidget
    fig = go.FigureWidget()

    # Add scatter traces for each cluster
    for cluster in unique_clusters:
        cluster_df = df[df["cluster"] == cluster]
        fig.add_scatter(
            x=cluster_df["T (K)"],
            y=cluster_df["H (A/m)"],
            mode="markers",
            marker=dict(color=color_map[cluster], size=10),
            name=f"Cluster {cluster}",
            customdata=cluster_df.index,  # store index to lookup image
            hovertemplate="T: %{x}<br>H: %{y}<extra></extra>",
        )

    # Add empty image on the right
    fig.update_layout(
        width=900,
        height=600,
        margin=dict(l=10, r=10, t=25, b=25),
        xaxis_title="T (K)",
        yaxis_title="H (A/m)",
        images=[
            dict(
                source="",  # placeholder
                xref="paper",
                yref="paper",
                x=1,  # just outside right of plot
                y=1,
                xanchor="right",
                yanchor="top",
                sizex=0.5,
                sizey=0.5,
                layer="above",
            )
        ],
    )

    # Update image on hover
    def update_image(trace, points, state):
        if points.point_inds:
            idx = points.point_inds[0]
            df_idx = trace.customdata[idx]
            if df_idx in image_cache:
                fig.layout.images[0].source = image_cache[df_idx]

    for trace in fig.data:
        trace.on_hover(update_image)

    return fig


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
    x_dim, y_dim, z_dim, _ = (
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
