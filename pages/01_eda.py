import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input
import plotly.express as px

from utils.data_loader import load_raw_data

dash.register_page(__name__, path="/eda")

df = load_raw_data()

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

numeric_cols     = [c for c in df.select_dtypes(include="number").columns if c != "Patient ID"]
categorical_cols = df.select_dtypes(include=["category","object"]).columns.tolist()
all_cols         = numeric_cols + categorical_cols

fig_corr = px.imshow(
    df[numeric_cols].corr(),
    title="Correlation Matrix"
)

missing = df.isna().sum().sort_values(ascending=True)
fig_missing = px.bar(
    x=missing.values,
    y=missing.index,
    orientation="h",
    title="Missing Values per Variable",
    labels={"x": "Number of Missing Values", "y": "Variable"}
)

layout = html.Div([
    html.H2("EDA: Raw Cancer Data"),
    html.Hr(),

    dbc.Row([
        dbc.Col(
            dcc.Graph(id="corr-matrix", figure=fig_corr),
            width=6
        ),
        dbc.Col(
            dcc.Graph(id="missing-plot", figure=fig_missing),
            width=6
        ),
    ], align="start", className="mb-4"),

    dbc.Row([

        dbc.Col([
            html.H4("Explore a single variable"),
            dcc.Dropdown(
                id="eda-feature",
                options=[{"label": c, "value": c} for c in all_cols],
                value=all_cols[0],
                clearable=False
            ),
        ], width=4),

        dbc.Col(
            dcc.Graph(id="dist-plot"),
            width=8
        ),
    ], align="start"),
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

