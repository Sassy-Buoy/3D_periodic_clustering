import base64
import os
from functools import lru_cache

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

# ---- Load data ----
@lru_cache(maxsize=1)
def load_dataset():
    """Cache dataset loading to avoid repeated file I/O."""
    csv_path = "data/dataset.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=["T (K)", "H (A/m)", "cluster_ae", "cluster_vae"])


df = load_dataset()

# ---- Predefined clustering metrics ----
CLUSTER_SCORES = {
    "cluster_ae": {"ari": 0.6230, "purity": 0.8812},
    "cluster_vae": {"ari": 0.8093, "purity": 0.9234},
}


# ---- Helper ----
def build_figure(cluster_col: str) -> go.Figure:
    """Build scatter figure for the selected clustering column."""
    if df.empty or cluster_col not in df.columns:
        return go.Figure()

    clusters = df[cluster_col].unique()
    colors = px.colors.qualitative.Dark24
    color_map = {c: colors[i % len(colors)] for i, c in enumerate(clusters)}

    fig = go.Figure()
    for cluster in clusters:
        cluster_df = df[df[cluster_col] == cluster]
        fig.add_scattergl(
            x=cluster_df["T (K)"],
            y=cluster_df["H (A/m)"],
            mode="markers",
            marker=dict(color=color_map[cluster], size=6, opacity=0.8),
            name=f"{cluster_col}: {cluster}",
            customdata=[[i] for i in cluster_df.index],
            hovertemplate="T: %{x}<br>H: %{y}<br>Index: %{customdata[0]}<extra></extra>",
        )

    fig.update_layout(
        width=600,
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="T (K)",
        yaxis_title="H (A/m)",
        template="plotly_white",
        showlegend=True,
    )
    return fig


# ---- Dash Page ----
dash.register_page(__name__, path="/clustering", name="Clustering results")

layout = html.Div(
    [
        html.H2("Magnetic Field Strength (A/m) vs Temperature (K)", style={"textAlign": "center"}),

        # Clustering type selector
        html.Div(
            [
                html.Label("Select Clustering Type:"),
                dcc.Dropdown(
                    id="cluster-type",
                    options=[
                        {"label": "Autoencoder Clustering", "value": "cluster_ae"},
                        {"label": "Variational Autoencoder Clustering", "value": "cluster_vae"},
                    ],
                    value="cluster_ae",
                    clearable=False,
                    style={"width": "300px"},
                ),
            ],
            style={"textAlign": "center", "marginBottom": "20px"},
        ),

        html.Div(
            [
                dcc.Graph(id="scatter", style={"width": "65%", "display": "inline-block"}),
                html.Img(
                    id="click-image",
                    style={"width": "30%", "display": "inline-block", "marginLeft": "2%"},
                ),
            ]
        ),

        html.Div(
            [
                html.P(id="ari-score"),
                html.P(id="purity-score"),
            ],
            style={"textAlign": "left"},
        ),

        html.P(
            "Click a point in the scatter plot to see the center z-slice of the corresponding simulation.",
            style={"textAlign": "center"},
        ),
    ],
    style={"maxWidth": "1000px", "margin": "auto"},
)


# ---- Image cache ----
@lru_cache(maxsize=256)
def get_encoded_image(df_idx: int) -> str:
    """Return base64-encoded image string with caching."""
    img_path = f"data/center_plots/{df_idx}.webp"
    if not os.path.exists(img_path):
        return blank_image()
    try:
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/webp;base64,{encoded}"
    except Exception:
        return blank_image()


def blank_image() -> str:
    """Return transparent 1x1 PNG."""
    return (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFhAJ/"
        "wlseKgAAAABJRU5ErkJggg=="
    )


# ---- Callbacks ----
@dash.callback(
    Output("scatter", "figure"),
    Output("ari-score", "children"),
    Output("purity-score", "children"),
    Input("cluster-type", "value"),
)
def update_scatter_and_scores(cluster_type):
    """Update scatter plot and evaluation scores."""
    fig = build_figure(cluster_type)
    scores = CLUSTER_SCORES.get(cluster_type, {"ari": 0.0, "purity": 0.0})
    ari_text = f"Adjusted Rand Index: {scores['ari']:.4f}"
    purity_text = f"Purity Score: {scores['purity']:.4f}"
    return fig, ari_text, purity_text


@dash.callback(
    Output("click-image", "src"),
    Input("scatter", "clickData"),
)
def update_image(clickData):
    """Update image on point click."""
    if not clickData:
        return blank_image()
    try:
        df_idx = clickData["points"][0].get("customdata", [None])[0]
        if df_idx is not None:
            return get_encoded_image(df_idx)
    except Exception:
        pass
    return blank_image()
