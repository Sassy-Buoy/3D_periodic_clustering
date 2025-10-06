import base64
import os

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
    fig.add_scatter(
        x=cluster_df["T (K)"],
        y=cluster_df["H (A/m)"],
        mode="markers",
        marker=dict(color=color_map[cluster], size=10),
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
dash.register_page(__name__, path="/page1")

layout = html.Div(
    [
        html.H2("H vs T with Hover Image"),
        html.Div(
            [
                dcc.Graph(
                    id="scatter",
                    figure=fig,
                    style={"width": "65%", "display": "inline-block"},
                ),
                html.Img(
                    id="hover-image",
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


# ---- Callback ----
@callback(Output("hover-image", "src"), Input("scatter", "hoverData"))
def update_image(hoverData):
    if hoverData:
        try:
            # Attempt to access customdata
            df_idx = hoverData["points"][0].get("customdata", [None])[0]
            if df_idx is not None:
                img_path = f"dash_app_viz/center_plots/{df_idx}.png"
                if os.path.exists(img_path):
                    return (
                        "data:image/png;base64,"
                        + base64.b64encode(open(img_path, "rb").read()).decode()
                    )
        except Exception as e:
            print("Error accessing customdata:", e)
    return ""  # Show nothing if no hover or no image
