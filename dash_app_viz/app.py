import dash
from dash import Dash, html
import os

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(
    [
        dash.page_container,
    ]
)

# Optimize for production
if os.environ.get("RENDER"):
    app.run_server(host="0.0.0.0", port=int(os.environ.get("PORT", 8050)), debug=False)
else:
    if __name__ == "__main__":
        app.run(host="0.0.0.0", port=8050, debug=True)
