import dash
import dash_bootstrap_components as dbc

app = dash.Dash(
  __name__,
  external_stylesheets=[dbc.themes.BOOTSTRAP],
  use_pages=True,
  pages_folder="pages"
)
server = app.server

app.layout = dbc.Container([
    dbc.NavbarSimple(
        children=[
            dbc.NavLink("EDA", href="/eda", active="exact"),
            dbc.NavLink("Classification", href="/classification", active="exact"),
            dbc.NavLink("Clustering", href="/clustering", active="exact"),
        ],
        brand="Cancer-Dash",
        color="primary", dark=True
    ),
    dash.page_container
], fluid=True)

if __name__ == "__main__":
    app.run(debug=True)
