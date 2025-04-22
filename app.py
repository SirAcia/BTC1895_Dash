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
            dbc.NavLink("Home", href="/", active="exact"),
            dbc.NavLink("EDA", href="/eda", active="exact"),
            dbc.NavLink("Classification Modelling", href="/classification", active="exact"),
            dbc.NavLink("Clustering Modelling", href="/clustering", active="exact"),
        ],
        brand="Cancer Classification & Clustering",
        brand_href="/",
        style={"backgroundColor": "#2A9D8F"}, 
        dark=True
    ),
    dash.page_container
], fluid=True)

if __name__ == "__main__":
    app.run(debug=True)
