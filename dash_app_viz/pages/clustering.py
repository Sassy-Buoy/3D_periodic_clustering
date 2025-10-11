import base64
from functools import lru_cache

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

# ---- Load data ----
# load dataset from csv
df = pd.read_csv("dataset.csv")

# ---- Build Plotly figure ----
# Assign colors to clusters
unique_clusters = df["cluster"].unique()
colors = px.colors.qualitative.Dark24
color_map = {
    cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)
}
df["color"] = df["cluster"].map(color_map)


fig = go.Figure()
for cluster in unique_clusters:
    cluster_df = df[df["cluster"] == cluster]
    fig.add_scattergl(
        x=cluster_df["T (K)"],
        y=cluster_df["H (A/m)"],
        mode="markers",
        marker=dict(color=color_map[cluster], size=6),
        name=f"Cluster {cluster}",
        customdata=[[i] for i in cluster_df.index],  # store index to lookup image
        hovertemplate="T: %{x}<br>H: %{y}<br>Index: %{customdata[0]}<extra></extra>",
    )

fig.update_layout(
    width=600,
    height=500,
    margin=dict(l=40, r=40, t=40, b=40),
    xaxis_title="T (K)",
    yaxis_title="H (A/m)",
)

# ---- Dash Page ----
dash.register_page(__name__, path="/clustering", name="Clustering results")

layout = html.Div(
    [
        html.H2("Magnetic field strength (A/m) vs Temperature (K)"),
        html.Div(
            [
                dcc.Graph(
                    id="scatter",
                    figure=fig,
                    style={"width": "65%", "display": "inline-block"},
                ),
                html.Img(
                    id="click-image",
                    style={
                        "width": "30%",
                        "display": "inline-block",
                        "margin-left": "2%",
                    },
                ),
            ]
        ),
    ]
)


# ---- Image cache ----
@lru_cache(maxsize=256)
def get_encoded_image(df_idx):
    """
    Cached base64 image loader.
    Converts images to base64 once and reuses them.
    """
    with open(f"center_plots/{df_idx}.webp", "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
        mime = "image/webp"
        return f"data:{mime};base64,{encoded}"
    return ""


# ---- Callback ----
@callback(Output("click-image", "src"), Input("scatter", "clickData"))
def update_image(clickData):
    if clickData:
        try:
            df_idx = clickData["points"][0].get("customdata", [None])[0]
            if df_idx is not None:
                return get_encoded_image(df_idx)
        except Exception as e:
            print("Error accessing customdata:", e)
    # if nothing is clicked, prompt user to click on a point
    return "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+P+/HgAFhAJ/wlseKgAAAABJRU5ErkJggg=="
