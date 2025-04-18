# pages/03_clustering.py

import dash
from dash import html, dcc, callback, Output, Input
import plotly.express as px
import pandas as pd
from sklearn.decomposition import PCA
from utils.data_loader import load_raw_data, preprocess_df
from utils.model_utils import fit_kmeans

dash.register_page(__name__, path="/clustering")

# 1) Load & preprocess once
df_raw = load_raw_data()
X, y   = preprocess_df(df_raw)

# 2) Page layout
layout = html.Div([
    html.H2("K‑Means Clustering Overview"),
    html.Label("Select number of clusters:"),
    dcc.Slider(
        id="n-clusters",
        min=2, max=10, step=1, value=2,
        marks={i: str(i) for i in range(2, 11)}
    ),
    html.H3("PCA 2D Projection"),
    dcc.Graph(id="pca-cluster-scatter"),
    html.H3("Cluster Feature Means (Scaled)"),
    dcc.Graph(id="cluster-profiles"),
], style={"padding": "1rem"})


# 3) Callback to update both graphs
@callback(
    Output("pca-cluster-scatter", "figure"),
    Output("cluster-profiles",      "figure"),
    Input("n-clusters",            "value")
)
def update_cluster_views(k):
    # --- run k-means ---
    km, labels = fit_kmeans(X, n_clusters=k)
    
    # --- PCA projection for 2D scatter ---
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)
    df_pca = pd.DataFrame({
        "PC1": pcs[:, 0],
        "PC2": pcs[:, 1],
        "Cluster": labels.astype(str),
        "Cancer Status": y.astype(str)
    })
    fig_pca = px.scatter(
        df_pca,
        x="PC1", y="PC2",
        color="Cluster", symbol="Cancer Status",
        title=f"PCA projection (k={k})",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"}
    )
    # show centroids in PC space
    centers_pca = pca.transform(km.cluster_centers_)
    fig_pca.add_scatter(
        x=centers_pca[:, 0], y=centers_pca[:, 1],
        mode="markers", marker_symbol="x", marker_size=12,
        marker_color="black", name="Centroids"
    )

    # --- Cluster feature‐mean profiles ---
    df_prof = X.copy()
    df_prof["Cluster"] = labels
    means = df_prof.groupby("Cluster").mean().reset_index()
    df_melt = means.melt(
        id_vars="Cluster",
        var_name="Feature",
        value_name="Mean"
    )
    fig_prof = px.bar(
        df_melt,
        x="Feature", y="Mean",
        facet_col="Cluster",
        title=f"Cluster Feature Means (k={k})",
        labels={"Mean": "Average (scaled)", "Feature": ""}
    )
    fig_prof.update_layout(showlegend=False)
    fig_prof.update_xaxes(tickangle=45)

    return fig_pca, fig_prof
