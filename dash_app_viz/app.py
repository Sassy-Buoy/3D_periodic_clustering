import os
import base64
import dash
from dash import dcc, html, Output, Input
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# load dataset from csv
df = pd.read_csv("dask_app_viz/dataset.csv")

# Assign colors to clusters
unique_clusters = df["cluster"].unique()
colors = px.colors.qualitative.Dark24
color_map = {cluster: colors[i % len(colors)] for i, cluster in enumerate(unique_clusters)}
df["color"] = df["cluster"].map(color_map)

# Build Plotly figure
fig = go.Figure()
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

fig.update_layout(
    width=600,
    height=500,
    margin=dict(l=40, r=40, t=40, b=40),
    xaxis_title="T (K)",
    yaxis_title="H (A/m)"
)

# ---- Dash App ----
app = dash.Dash(__name__)
server = app.server  # needed for deployment

app.layout = html.Div([
    html.H2("H vs T with Hover Image"),
    html.Div([
        dcc.Graph(id="scatter", figure=fig, style={"width": "65%", "display": "inline-block"}),
        html.Img(id="hover-image", style={"width": "30%", "display": "inline-block", "margin-left": "2%"})
    ])
])

@app.callback(
    Output("hover-image", "src"),
    Input("scatter", "hoverData")
)
def update_image(hoverData):
    if hoverData:
        df_idx = hoverData["points"][0]["customdata"]
        img_path = f"center_plots/{df_idx}.png"
        if os.path.exists(img_path):
            return "data:image/png;base64," + base64.b64encode(open(img_path, "rb").read()).decode()
    return ""  # show nothing if no hover or no image

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=8050, debug=True)
