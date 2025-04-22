# app.py 

# libaries 
import dash
import dash_bootstrap_components as dbc

# app set up 
app = dash.Dash(
  __name__,
  external_stylesheets=[dbc.themes.BOOTSTRAP],
  use_pages=True, # dash uses pages for structure
  pages_folder="pages" # make sure to keep pages in pages folder in repo
)

# server run line
server = app.server

# navigation layout (container for nav bar)
app.layout = dbc.Container([

    # nav bar set up 
    dbc.NavbarSimple(

        children=[
            # nav bar buttons & respective pages 
            dbc.NavLink("Summary", href="/", active="exact"),
            dbc.NavLink("EDA", href="/eda", active="exact"),
            dbc.NavLink("Classification Modelling", href="/classification", active="exact"),
            dbc.NavLink("Clustering Modelling", href="/clustering", active="exact"),
        ],
        
        # formatting title, color, etc. 
        brand="Cancer Classification & Clustering",
        brand_href="/",
        style={"backgroundColor": "#2A9D8F"}, 
        dark=True
    ),
    dash.page_container
], fluid=True)

if __name__ == "__main__":
    app.run(debug=True)
