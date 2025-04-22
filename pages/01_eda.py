# EDA page

# libraries/imports 
import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Output, Input
import plotly.express as px
from utils.data_loader import load_raw_data

# registering page for directory
dash.register_page(__name__, path="/eda")

# loading raw data from utility page (back end so only processes data once) 
df = load_raw_data()

# renaming columns for readability
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

# getting numeric variables and categorical variables (minus patient ID) for distributions
# getting numeric vairiables by interating over df 
numeric_cols     = [col for col in df.select_dtypes(include="number").columns if col != "Patient ID"]
categorical_cols = df.select_dtypes(include=["category","object"]).columns.tolist()
all_cols         = numeric_cols + categorical_cols

# setting color palette for continous color scale
continuous_palette = px.colors.sequential.Plasma_r

# creating correlation matrix of variables
fig_corr = px.imshow(
    df[numeric_cols].corr(),
    title="Correlation Matrix of Synthetic Data",
    color_continuous_scale=continuous_palette
)

# formatting tooltip 
fig_corr.update_traces(
    hovertemplate=
      "Corr(%{x}, %{y}) = %{z:.2f}" + 
      "<extra></extra>"   # drop the secondary box
)

# calculating missingness 
missing = df.isna().sum().sort_values(ascending=True)

# creating df for graphing 
missing_df = (
    missing
    .reset_index(name="Missing Count")
    .rename(columns={"index": "Variable"})
)

# creating bar graph for missingness 
fig_missing = px.bar(
    missing_df,
    x="Missing Count",
    y="Variable",
    orientation="h",
    title="Missing Values per Variable",
     labels={"MissingCount": "Number of Missing Values", "Variable": "Variable"},
    color="Variable",
    color_discrete_sequence=px.colors.qualitative.Pastel # using different color palette for non-continous color sequence 
)

# formatting tooltip
fig_missing.update_traces(
    hovertemplate="Variable: %{y}<br>Missing: %{x:d}<extra></extra>"
)

# removing legend
fig_missing.update_layout(showlegend=False)

# defining page layout
layout = html.Div([
    # first portion (explanation/intro)
    html.H2("EDA: Raw Cancer Data"),
    html.P(
        "The data utilised in this analysis is comprised of a synthetic dataset of 1000, patient-level observations. "
        "Data includes 16 variables ranging across different areas including clinical & demographic data "
        "(including age, sex, smoking status, and family history), genetic data (including TP53, BRCA1, KRAS, and total " 
        "mutations),  biomarkers and blood test data (including white blood cell count as well as CEA, AFP, and CRP levels), "
        "imaging data (including tumour size, location, and density), and cancer status (presence/absence). "
    ),

        html.P(
        "Overall, there are no major amounts of missingness or obvious correlation (see graphs below), with the largest amount " 
        "of missingness seen in some genomic and biomarker data. Regardless, the highest amount seen is well below a 30% threshold. "
    ),

    # splitting into 2 columns for first portion (pairing correlation matrix and missing plot)
    html.Hr(),

    # first row as columns, one for each graph 
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

    # second row for individual variable distribution
    html.P(
        "Click on a variable to explore it's distrbution below. Generally, all variables largely follow a normal or uniform distribution. " 
    ),

    # same thing for second portion, one row, split into 2 columns
    dbc.Row([
        # setting up dropdown menu
        dbc.Col([
            html.H4("Explore a single variable"),
            dcc.Dropdown(
            id="eda-feature",
            options=[
                {"label": c, "value": c}
                for c in all_cols
                if c not in ("Cancer Status", "Patient ID")
                    ],
            value=all_cols[0],
            clearable=False
            ),
        ], width=4),

        # second column is distribution plot 
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

# function for updating distribution plot 
def update_dist(col):
    # setting color palette 
    palette = px.colors.qualitative.Pastel
    
    # iterating over each column to get distributions
    if col in numeric_cols:
        # for numeric variables, generating histogram
        fig = px.histogram(
            df,
            x=col,
            color="Cancer Status",
            nbins=30,
            title=f"Distribution of {col} by Cancer Status",
            hover_data={
                col: ":.2f",              
                "Cancer Status": True,               
            },
            color_discrete_sequence=palette
        )

        fig.update_layout(hovermode="x unified")

    else:
        # else create a bar graph for categorical variables
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
                "Cancer Status": True    
            },
            color_discrete_sequence=palette
        )

        fig.update_layout(hovermode="closest")

    return fig

