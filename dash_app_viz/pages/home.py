import dash
from dash import html, dcc

dash.register_page(
    __name__,
    path="/",  # root URL
    name="Home",
    title="3D Periodic Clustering - Home",
)

layout = html.Div(
    [html.Div(
                            [
                                html.H1(
                                    "ML-based Clustering and Dimensionality Reduction for 3D Magnetic Field Simulations",
                                    style={
                                        "textAlign": "center",
                                        "color": "#2c3e50",
                                        "marginBottom": "10px",
                                    },
                                ),
                            ],
                            style={
                                "padding": "20px",
                                "backgroundColor": "#ecf0f1",
                                "marginBottom": "30px",
                            },
                        ),
        html.Div(
            [
                # Left half: Header, Project Overview, Data Summary
                html.Div(
                    [
                        # Header
                        
                        # Project Overview Section
                        html.Div(
                            [
                                html.H3(
                                    "🔬 Key Features:",
                                    style={"color": "#34495e", "marginBottom": "15px"},
                                ),
                                html.Ul(
                                    [
                                        html.Li(
                                            "Dimensionality reduction using Variational Autoencoders and Autoencoders"
                                        ),
                                        html.Li(
                                            "Clustering of latent space using Gaussian Mixture Models"
                                        ),
                                        html.Li("Latent Space Visualization using UMAP"),
                                    ],
                                    style={"fontSize": "14px", "lineHeight": "1.8"},
                                ),
                            ],
                            style={"marginBottom": "40px"},
                        ),
                        # Data Summary Section
                        html.Div(
                            [
                                html.H2(
                                    "📊 Data Summary",
                                    style={
                                        "color": "#2c3e50",
                                        "borderBottom": "2px solid #e74c3c",
                                        "paddingBottom": "10px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                html.H3(
                                                    2601,
                                                    style={
                                                        "fontSize": "24px",
                                                        "color": "#e74c3c",
                                                        "margin": "0",
                                                    },
                                                ),
                                                html.P(
                                                    "Total Simulations",
                                                    style={
                                                        "fontSize": "14px",
                                                        "color": "#7f8c8d",
                                                        "margin": "5px 0",
                                                    },
                                                ),
                                            ],
                                            className="stat-box",
                                            style={
                                                "textAlign": "center",
                                                "backgroundColor": "#fff",
                                                "padding": "20px",
                                                "borderRadius": "8px",
                                                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                                "width": "22%",
                                                "display": "inline-block",
                                                "margin": "0 1.5%",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.H3(
                                                    "0 - 3000 K",
                                                    style={
                                                        "fontSize": "24px",
                                                        "color": "#f39c12",
                                                        "margin": "0",
                                                    },
                                                ),
                                                html.P(
                                                    "Temperature Range",
                                                    style={
                                                        "fontSize": "14px",
                                                        "color": "#7f8c8d",
                                                        "margin": "5px 0",
                                                    },
                                                ),
                                            ],
                                            className="stat-box",
                                            style={
                                                "textAlign": "center",
                                                "backgroundColor": "#fff",
                                                "padding": "20px",
                                                "borderRadius": "8px",
                                                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                                "width": "22%",
                                                "display": "inline-block",
                                                "margin": "0 1.5%",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.H3(
                                                    "0 - 0.39e6 A/m",
                                                    style={
                                                        "fontSize": "24px",
                                                        "color": "#27ae60",
                                                        "margin": "0",
                                                    },
                                                ),
                                                html.P(
                                                    "Magnetic Field Range",
                                                    style={
                                                        "fontSize": "14px",
                                                        "color": "#7f8c8d",
                                                        "margin": "5px 0",
                                                    },
                                                ),
                                            ],
                                            className="stat-box",
                                            style={
                                                "textAlign": "center",
                                                "backgroundColor": "#fff",
                                                "padding": "20px",
                                                "borderRadius": "8px",
                                                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                                "width": "22%",
                                                "display": "inline-block",
                                                "margin": "0 1.5%",
                                            },
                                        ),
                                        html.Div(
                                            [
                                                html.H3(
                                                    "80x80x80x3 --> 6",
                                                    style={
                                                        "fontSize": "24px",
                                                        "color": "#2980b9",
                                                        "margin": "0",
                                                    },
                                                ),
                                                html.P(
                                                    "Dimensionality",
                                                    style={
                                                        "fontSize": "14px",
                                                        "color": "#7f8c8d",
                                                        "margin": "5px 0",
                                                    },
                                                ),
                                            ],
                                            className="stat-box",
                                            style={
                                                "textAlign": "center",
                                                "backgroundColor": "#fff",
                                                "padding": "20px",
                                                "borderRadius": "8px",
                                                "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                                "width": "22%",
                                                "display": "inline-block",
                                                "margin": "0 1.5%",
                                            },
                                        ),
                                    ],
                                    style={"marginBottom": "30px"},
                                ),
                            ],
                            style={"marginBottom": "40px"},
                        ),
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "paddingRight": "20px",
                        "boxSizing": "border-box",
                    },
                ),
                # Right half: Navigation Section
                html.Div(
                    [
                        html.H2(
                            "🎯 Explore the Results",
                            style={
                                "color": "#2c3e50",
                                "borderBottom": "2px solid #9b59b6",
                                "paddingBottom": "10px",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.H3(
                                            "🔄 Reconstruction Quality",
                                            style={
                                                "color": "#e67e22",
                                                "marginBottom": "15px",
                                            },
                                        ),
                                        html.P(
                                            "Compare reconstruction quality between Variational Autoencoders and standard Autoencoders. Examine how well each model captures the original magnetic field structures.",
                                            style={
                                                "fontSize": "14px",
                                                "lineHeight": "1.6",
                                                "marginBottom": "15px",
                                            },
                                        ),
                                        dcc.Link(
                                            html.Button(
                                                "View Reconstructions",
                                                style={
                                                    "backgroundColor": "#e67e22",
                                                    "color": "white",
                                                    "border": "none",
                                                    "padding": "10px 20px",
                                                    "borderRadius": "5px",
                                                    "cursor": "pointer",
                                                    "fontSize": "14px",
                                                },
                                            ),
                                            href="/reconstruction",
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": "#fff",
                                        "padding": "25px",
                                        "borderRadius": "8px",
                                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                        "width": "100%",
                                        "marginBottom": "20px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.H3(
                                            "🔍 Clustering Analysis",
                                            style={
                                                "color": "#8e44ad",
                                                "marginBottom": "15px",
                                            },
                                        ),
                                        html.P(
                                            "Explore the clustering results in Magnetic field strength-Temperature parameter space. In order to evaluate clustering quality, 10% of the data points were labeled manually.",
                                            style={
                                                "fontSize": "14px",
                                                "lineHeight": "1.6",
                                                "marginBottom": "15px",
                                            },
                                        ),
                                        dcc.Link(
                                            html.Button(
                                                "View Clustering Results",
                                                style={
                                                    "backgroundColor": "#8e44ad",
                                                    "color": "white",
                                                    "border": "none",
                                                    "padding": "10px 20px",
                                                    "borderRadius": "5px",
                                                    "cursor": "pointer",
                                                    "fontSize": "14px",
                                                },
                                            ),
                                            href="/clustering",
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": "#fff",
                                        "padding": "25px",
                                        "borderRadius": "8px",
                                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                        "width": "100%",
                                        "marginBottom": "20px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.H3(
                                            "🧩 Latent Space Visualization",
                                            style={
                                                "color": "#16a085",
                                                "marginBottom": "15px",
                                            },
                                        ),
                                        html.P(
                                            "Visualize the latent space using UMAP to understand how the models encode the data throughout the training process. Compare the latent spaces of Variational Autoencoders and standard Autoencoders.",
                                            style={
                                                "fontSize": "14px",
                                                "lineHeight": "1.6",
                                                "marginBottom": "15px",
                                            },
                                        ),
                                        dcc.Link(
                                            html.Button(
                                                "View Latent Space",
                                                style={
                                                    "backgroundColor": "#16a085",
                                                    "color": "white",
                                                    "border": "none",
                                                    "padding": "10px 20px",
                                                    "borderRadius": "5px",
                                                    "cursor": "pointer",
                                                    "fontSize": "14px",
                                                },
                                            ),
                                            href="/latent-space",
                                        ),
                                    ],
                                    style={
                                        "backgroundColor": "#fff",
                                        "padding": "25px",
                                        "borderRadius": "8px",
                                        "boxShadow": "0 2px 4px rgba(0,0,0,0.1)",
                                        "width": "100%",
                                    },
                                ),
                            ],
                            style={"marginBottom": "30px"},
                        ),
                    ],
                    style={
                        "width": "50%",
                        "display": "inline-block",
                        "verticalAlign": "top",
                        "paddingLeft": "20px",
                        "boxSizing": "border-box",
                    },
                ),
            ],
            style={
                "maxWidth": "1200px",
                "margin": "0 auto",
                "padding": "0 20px",
                "display": "flex",
                "flexDirection": "row",
                "justifyContent": "space-between",
            },
        ),
    ],
    style={
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f8f9fa",
        "minHeight": "100vh",
        "padding": "0",
    },
)
