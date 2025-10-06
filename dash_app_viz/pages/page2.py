import base64
import io

import dash
import discretisedfield as df
import matplotlib.pyplot as plt
import torch
from dash import Input, Output, dcc, html

# ---- Load data ----
example_sims = torch.load("dash_app_viz/example_sims.pt")  # Original simulations
ae_recon = torch.load("dash_app_viz/vanilla_reconstruction.pt")  # Autoencoder
vae_recon = torch.load("dash_app_viz/variational_reconstruction.pt")  # VAE


# ---- Helper: tensor → 2D mpl plot for a slice at z_value ----
def plot_field_xy_from_tensor(tensor, z_value):
    """Return a Matplotlib plot of a 2D slice as a base64 PNG for Dash."""
    if tensor.requires_grad:
        tensor = tensor.detach()

    v_dim, x_dim, y_dim, z_dim = tensor.shape
    value = tensor.cpu().numpy().transpose((1, 2, 3, 0))

    mesh = df.Mesh(
        p1=(0, 0, 0),
        p2=(x_dim - 1, y_dim - 1, z_dim - 1),
        n=(x_dim, y_dim, z_dim),
    )

    field = df.Field(mesh=mesh, value=value, nvdim=v_dim)

    # Create a figure and axes manually
    fig, ax = plt.subplots(figsize=(5, 5))

    # Plot slice on the axes
    field.sel(z=z_value).mpl(ax=ax)

    # Save figure to PNG in memory
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    # Encode PNG as base64
    img_base64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{img_base64}"


# ---- Dash Page ----
dash.register_page(__name__, path="/page2")

layout = html.Div(
    [
        html.H2("Simulation Reconstruction Comparison", style={"textAlign": "center"}),
        # Model Type Selection
        html.Div(
            [
                html.Label("Model Type"),
                dcc.RadioItems(
                    id="model-type",
                    options=[
                        {"label": "Autoencoder", "value": "autoencoder"},
                        {"label": "Variational Autoencoder", "value": "vae"},
                    ],
                    value="autoencoder",
                    inline=True,
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        # Simulation Index Selection
        html.Div(
            [
                html.Label("Simulation Index"),
                dcc.Dropdown(
                    id="sim-index",
                    options=[
                        {"label": f"Simulation {i + 1}", "value": i}
                        for i in range(len(example_sims))
                    ],
                    value=0,
                    style={"width": "250px"},
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        # Z Value Slider
        html.Div(
            [
                html.Label("Z Value"),
                dcc.Slider(
                    id="z-value",
                    min=0,
                    max=48,
                    step=None,  # Smooth continuous slider
                    value=24,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
            style={"marginBottom": "30px"},
        ),
        # Plot Container
        html.Div(
            [
                html.Label("Comparison"),
                html.Div(id="comparison-plot-container"),
            ]
        ),
    ],
    style={"maxWidth": "900px", "margin": "auto"},
)


# ---- Callback ----
@dash.callback(
    Output("comparison-plot-container", "children"),
    [
        Input("model-type", "value"),
        Input("sim-index", "value"),
        Input("z-value", "value"),
    ],
)
def update_graphs(model_type, sim_index, z_value):
    """Update both plots when user changes model, sim index, or z value."""
    # Select simulation data
    original_sim = example_sims[sim_index]
    reconstructed_sim = (
        ae_recon[sim_index] if model_type == "autoencoder" else vae_recon[sim_index]
    )

    # Generate the Matplotlib plots
    original_plot = plot_field_xy_from_tensor(original_sim, z_value)
    reconstructed_plot = plot_field_xy_from_tensor(reconstructed_sim, z_value)

    # Display side by side
    return html.Div(
        [
            html.Img(
                src=original_plot, style={"width": "48%", "display": "inline-block"}
            ),
            html.Img(
                src=reconstructed_plot,
                style={"width": "48%", "display": "inline-block"},
            ),
        ]
    )
