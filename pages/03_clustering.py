# Clustering page

# libraries/imports 
import dash
from dash import html, dcc, callback, Output, Input
import plotly.express as px
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from utils.data_loader import load_raw_data, preprocess_df
from utils.model_utils import fit_kmeans

# registering page for directory
dash.register_page(__name__, path="/clustering")

# loading data 
df_raw = load_raw_data()

# processing data 
X, y = preprocess_df(df_raw)

# renaming columns for readability
X.rename(columns={
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
}, inplace=True)

y = y.rename("Cancer Status")

# defining page layout
layout = html.Div([
    # first portion/intro
    html.H2("K‑Means Clustering Overview"),
    html.P(
        "Use this slider to select the number of clusters (k) applied to the cancer data. "
        "This visualization is a 3D approximation of the clustering applied to the data using " 
        "Principal Compoment Analysis (PCA) to view clustering in reduced dimensions, "
        "and the feature means chart summarizes each cluster's average feature values. " 
    ),

    # slider for clusters + clusters 
    html.Label("Select number of clusters:"),
    dcc.Slider(
        id="n-clusters",
        min=2, max=10, step=1, value=2,
        marks={i: str(i) for i in range(2, 11)},
        tooltip={"placement": "bottom"}
    ),
    html.H3("3D PCA Projection"),
    dcc.Graph(id="pca-cluster-scatter"),

    # cluster comparison portion
    html.H3("Cluster Feature Means (Scaled)"),
    html.P(
        "Below shows the feature distribution across all different clusters. If feature distributions " 
        "differ greatly between clusters, that is an indicator that k-means clustering is a optimal " 
        "method to identify characteristics associated with cancer. Thus far, no cluster shows significant " 
        "differences between one another."
    ), 
    dcc.Graph(id="cluster-profiles"),

], style={"padding": "1rem"})

# callback for detailed cluster representation + graphs
@callback(
    Output("pca-cluster-scatter", "figure"),
    Output("cluster-profiles", "figure"),
    Input("n-clusters", "value")
)
# function for 3D visualization of clustering + graphs
def update_cluster_views(k):
    try:
        # fits k means with data 
        km, labels = fit_kmeans(X, n_clusters=k)
        labels_str = labels.astype(str)

        # defining color palette 
        palette = px.colors.qualitative.Pastel

        # defining 3D PCA for visualization
        pca = PCA(n_components=3, random_state=42)
        pcs = pca.fit_transform(X)
        df_pca = pd.DataFrame(pcs, columns=["PC1", "PC2", "PC3"], index=X.index)
        df_pca["Cluster"] = labels_str
        df_pca["Cancer Status"] = y.astype(str)

        # creating 3D scatter plot for approx. visualization
        fig_pca3d = px.scatter_3d(
            df_pca, x="PC1", y="PC2", z="PC3",
            color="Cluster", symbol="Cancer Status",
            title=f"3D PCA Projection (k={k})",
            hover_data={
                "PC1":  ":.2f",
                "PC2":  ":.2f",
                "PC3":  ":.2f",
                "Cancer Status": True,   
                "Cluster": False         
            },
            color_discrete_sequence=palette # color is pastel (from custom palette)
        )

        # removing legend
        fig_pca3d.update_layout(showlegend=False)

        # making copy of data for clustering 
        df_prof = X.copy()

        # getting average number of points for each variable per cluster 
        df_prof["Cluster"] = labels_str
        means = df_prof.groupby("Cluster").mean().reset_index()
        df_melt2 = means.melt(id_vars="Cluster", var_name="Feature", value_name="Mean")

        # creating bar chart for feature distribution
        fig_prof = px.bar(
            df_melt2,
            x="Feature", y="Mean",
            facet_col="Cluster",
            color="Feature",
            title=f"Cluster Feature Means (k={k})",
            labels={"Mean": "Average (scaled)", "Feature": ""},
            hover_data={
                "Mean":":.2f",
                "Feature": True,   
                "Cluster": True   
            },
            color_discrete_sequence=palette # color is pastel (from custom palette)
        )

        fig_prof.update_layout(showlegend=False)
        fig_prof.update_xaxes(tickangle=45)

        return fig_pca3d, fig_prof

    except Exception as e:
        print("Error in callback:", e)
        raise dash.exceptions.PreventUpdate
