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
    title="Correlation Matrix of Synthetic Data"
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
    html.P(
        "The data utilised in this analysis is comprised of a synthetic dataset of 1000, patient-level observations. "
        "Data includes 16 variables ranging across different areas including clinical & demographic data "
        "(including age, sex, smoking status, and family history), genetic data (including TP53, BRCA1, KRAS, and total " 
        "mutations),  biomarkers and blood test data (including white blood cell count as well as CEA, AFP, and CRP levels), "
        "imaging data (including tumour size, location, and density), and cancer status (presence/absence). "
    ),
        html.P(
        "Overall, there are no major amounts of missingness or obvious correlation (see graphs below), with the largest amount" 
        "of missingness seen in some genomic and biomarker data. Regardless, the highest amount seen is well below a 30% threshold. "
    ),

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

    html.P(
        "Click on a variable to explore it's distrbution below. Generally, all variables largely follow a normal or uniform distribution. " 
    ),

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

    palette = px.colors.qualitative.Pastel
    
    if col in numeric_cols:
        # Numeric: histogram, colored by Cancer Status
        fig = px.histogram(
            df,
            x=col,
            color="Cancer Status",
            nbins=30,
            title=f"Distribution of {col} by Cancer Status",
            hover_data={
                col: ":.2f",              
                "Cancer Status": True,    
                "count": ":d"             
            },
            color_discrete_sequence=palette
        )

        fig.update_layout(hovermode="x unified")

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
            title=f"Counts of {col} by Cancer Status",
            hover_data={
                col: False,               
                "Count": ":d",           
                "Cancer Status": True    
            },
            color_discrete_sequence=palette
        )

        fig.update_layout(hovermode="closest")

    return fig

