import plotly.graph_objects as go
import pandas as pd
import pathlib as pl


def training_log():
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
    for i, path in enumerate(pl.Path("lightning_logs").glob("version_*")):
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
                #visible="legendonly",
            )
        )

        # Add validation loss
        fig.add_trace(
            go.Scatter(
                x=df["epoch"],
                y=df["val_loss"],
                mode="lines",
                name=f"val_loss_{i}",
                visible="legendonly",
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
