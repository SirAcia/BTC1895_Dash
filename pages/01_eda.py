
import dash
from dash import html, dcc, callback, Output, Input
import plotly.express as px
from utils.data_loader import load_raw_data

# 1) Register this file as the "/eda" page
dash.register_page(__name__, path="/eda")

# 2) Load the raw DataFrame once
df_raw = load_raw_data()

# 3) Figure out which cols are numeric vs categorical
numeric_cols     = df_raw.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = df_raw.select_dtypes(include=["category", "object"]).columns.tolist()
all_cols         = numeric_cols + categorical_cols

# 4) Page layout: dropdown + two graph placeholders
layout = html.Div([
    html.H2("EDA: Raw Cancer Data"),
    dcc.Dropdown(
        id="eda-feature",
        options=[{"label": c, "value": c} for c in all_cols],
        value=all_cols[0],    # default = first column
        clearable=False
    ),
    dcc.Graph(id="histogram"),
    dcc.Graph(id="scatterplot"),
], style={"padding": "1rem"})


# 5) Histogram callback: works for both numeric & categorical
@callback(
    Output("histogram", "figure"),
    Input("eda-feature", "value")
)
def update_histogram(col):
    fig = px.histogram(
        df_raw,
        x=col,
        title=f"Histogram of {col}",
        nbins=30 if col in numeric_cols else None
    )
    return fig


# 6) Scatter‐plot callback: branches on type
@callback(
    Output("scatterplot", "figure"),
    Input("eda-feature", "value")
)
def update_scatterplot(col):
    if col in numeric_cols:
        # scatter against the next numeric column
        other = next(c for c in numeric_cols if c != col)
        fig = px.scatter(
            df_raw,
            x=col, y=other,
            title=f"{col} vs. {other}"
        )
    else:
        # categorical: plot first two numerics, color by the category
        num1, num2 = numeric_cols[:2]
        fig = px.scatter(
            df_raw,
            x=num1, y=num2,
            color=col,
            title=f"{num1} vs. {num2} colored by {col}"
        )
    return fig
