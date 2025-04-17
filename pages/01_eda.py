import dash
from dash import html, dcc
import plotly.express as px
from utils.data_loader import load_raw_data, preprocess_df

dash.register_page(__name__, path="/eda")

# load & prep once
df_raw = load_raw_data()
X, y = preprocess_df(df_raw)
df = X.copy()
df["Cancer_Status"] = y

layout = html.Div([
    html.H2("EDA: Synthetic Cancer Data"),
    dcc.Dropdown(
        id="eda-feature",
        options=[{"label": c, "value": c} for c in X.columns],
        value=X.columns[0],
    ),
    dcc.Graph(id="hist"),
    dcc.Graph(id="scatter")
])

@dash.callback(
    dash.Output("hist", "figure"),
    dash.Input("eda-feature", "value")
)
def update_hist(col):
    fig = px.histogram(df, x=col, color="Cancer_Status",
                       title=f"Distribution of {col}")
    return fig

@dash.callback(
    dash.Output("scatter", "figure"),
    dash.Input("eda-feature", "value")
)
def update_scatter(col):
    other = [c for c in X.columns if c!=col][0]
    fig = px.scatter(df, x=col, y=other, color="Cancer_Status",
                     title=f"{col} vs. {other}")
    return fig
