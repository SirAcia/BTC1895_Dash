# pages/01_eda.py

import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input
import plotly.express as px

from utils.data_loader import load_raw_data

dash.register_page(__name__, path="/eda")

# 1) Load once
df = load_raw_data()

# 2) Rename columns on the existing DataFrame
df.rename(columns={
    "Patient_ID":      "Patient ID",
    "Smoking_Status":  "Smoking Status",
    "Family_History":  "Family History",
    "TP53_Mutation":   "TP53 Mutation",
    "BRCA1_Mutation":  "BRCA1 Mutation",
    "KRAS_Mutation":   "KRAS Mutation",
    "Total_Mutations": "Total Mutations",
    "CEA_Level":       "CEA Level",
    "AFP_Level":       "AFP Level",
    "WBC_Count":       "WBC Count",
    "CRP_Level":       "CRP Level",
    "Tumor_Size":      "Tumor Size",
    "Tumor_Location":  "Tumor Location",
    "Tumor_Density":   "Tumor Density",
    "Cancer_Status":   "Cancer Status",
}, inplace=True)

# 3) Identify numeric vs categorical
numeric_cols     = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(include=["category","object"]).columns.tolist()
all_cols         = numeric_cols + categorical_cols

# 4) Precompute key figures
fig_corr = px.imshow(
    df[numeric_cols].corr(),
    title="Correlation Matrix"
)
fig_pair = px.scatter_matrix(
    df,
    dimensions=numeric_cols[:4],
    color=categorical_cols[0] if categorical_cols else None,
    title="Pairwise Scatter (first 4 nums)"
)

# 5) Layout: sidebar + main
layout = html.Div([
    html.H2("EDA: Raw Cancer Data"),
    html.Hr(),
    dbc.Row([
        dbc.Col([
            html.H4("Explore a single variable"),
            dcc.Dropdown(
                id="eda-feature",
                options=[{"label": c, "value": c} for c in all_cols],
                value=all_cols[0],
                clearable=False
            ),
            dcc.Graph(id="dist-plot")
        ], width=4),

        dbc.Col([
            html.H4("Key summaries"),
            dcc.Graph(id="corr-matrix", figure=fig_corr),
            dcc.Graph(id="pair-matrix", figure=fig_pair)
        ], width=8),
    ], align="start")
], style={"padding": "1rem"})


@callback(
    Output("dist-plot", "figure"),
    Input("eda-feature", "value")
)
def update_dist(col):
    if col in numeric_cols:
        # Numeric: histogram, colored by Cancer Status
        fig = px.histogram(
            df,
            x=col,
            color="Cancer Status",
            nbins=30,
            title=f"Distribution of {col} by Cancer Status"
        )
    else:
        # Categorical: compute counts, then bar chart grouped by Cancer Status
        count_df = (
            df
            .groupby([col, "Cancer Status"])
            .size()
            .reset_index(name="Count")
        )
        fig = px.bar(
            count_df,
            x=col,
            y="Count",
            color="Cancer Status",
            barmode="group",
            title=f"Counts of {col} by Cancer Status"
        )
    return fig

