import base64
import io
from functools import lru_cache

import dash
import discretisedfield as df
import matplotlib.pyplot as plt
import numpy as np
from dash import Input, Output, dcc, html

# ---- Load data ----
example_sims = np.load("data/example_sims.npy", mmap_mode="r")
ae_recon = np.load("data/vanilla_reconstruction.npy", mmap_mode="r")
vae_recon = np.load("data/variational_reconstruction.npy", mmap_mode="r")


# ---- Mesh cache ----
_mesh_cache = {}
def get_mesh(shape):
    if shape not in _mesh_cache:
        x, y, z, _ = shape
        _mesh_cache[shape] = df.Mesh(p1=(0, 0, 0), p2=(x - 1, y - 1, z - 1), n=(x, y, z))
    return _mesh_cache[shape]


# ---- Cached slice rendering ----
@lru_cache(maxsize=256)
def plot_field_xy_from_tensor_cached(key: str, z_value: int) -> str:
    """Cached base64 PNG from tensor bytes (serialized key) and z index."""
    array = np.load(io.BytesIO(base64.b64decode(key)), allow_pickle=False)
    return plot_field_xy_from_tensor(array, z_value)


def plot_field_xy_from_tensor(array, z_value):
    """Return base64 PNG for a 2D z-slice of a 4D tensor."""
    x_dim, y_dim, z_dim, v_dim = array.shape
    mesh = get_mesh(array.shape)
    field = df.Field(mesh=mesh, value=array, nvdim=v_dim)

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    field.sel(z=z_value).mpl(ax=ax)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=80)
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('ascii')}"


# ---- Dash Page ----
dash.register_page(__name__, path="/reconstruction", name="Reconstruction results")

layout = html.Div(
    [
        html.H2("Simulation Reconstruction Comparison", style={"textAlign": "center"}),

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

        html.Div(
            [
                html.Label("Simulation Index"),
                dcc.Dropdown(
                    id="sim-index",
                    options=[{"label": f"Simulation {i+1}", "value": i} for i in range(len(example_sims))],
                    value=0,
                    style={"width": "250px"},
                ),
            ],
            style={"marginBottom": "20px"},
        ),

        html.Div(
            [
                html.Label("Z Value"),
                dcc.Slider(
                    id="z-value",
                    min=0,
                    max=47,
                    step=1,
                    value=24,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ],
            style={"marginBottom": "30px"},
        ),

        html.Div(
            [
                html.Label("Comparison"),
                dcc.Loading(
                    id="loading-container",
                    type="circle",
                    children=html.Div(id="comparison-plot-container"),
                ),
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
    prevent_initial_call=False,
)
def update_graphs(model_type, sim_index, z_value):
    """Render comparison plots."""
    recon = {"autoencoder": ae_recon, "vae": vae_recon}[model_type][sim_index]
    original_sim = example_sims[sim_index]

    original_plot = plot_field_xy_from_tensor(original_sim, z_value)
    reconstructed_plot = plot_field_xy_from_tensor(recon, z_value)

    return html.Div(
        [
            html.Div(
                [
                    html.H4("Original", style={"textAlign": "center", "margin": "5px 0"}),
                    html.Img(src=original_plot, style={"width": "100%"}),
                ],
                style={"width": "48%", "display": "inline-block", "verticalAlign": "top"},
            ),
            html.Div(
                [
                    html.H4("Reconstruction", style={"textAlign": "center", "margin": "5px 0"}),
                    html.Img(src=reconstructed_plot, style={"width": "100%"}),
                ],
                style={"width": "48%", "display": "inline-block", "verticalAlign": "top"},
            ),
        ]
    )
