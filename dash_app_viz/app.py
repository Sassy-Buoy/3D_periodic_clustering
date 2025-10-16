import dash
from dash import Dash, html

app = Dash(__name__, use_pages=True, suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div(
    [
        dash.page_container,
    ]
)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=True)
