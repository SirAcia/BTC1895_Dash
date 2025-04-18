# pages/03_clustering.py

import dash
from dash import html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
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
    html.H3("3D PCA Projection"),
    dcc.Graph(id="pca-cluster-scatter"),
    html.H3("PCA Component Distributions"),
    dcc.Graph(id="pca-histograms"),
    html.H3("Cluster Feature Means (Scaled)"),
    dcc.Graph(id="cluster-profiles"),
    html.H3("Cluster vs. Cancer Status Confusion Matrix"),
    dcc.Graph(id="confusion-matrix"),
], style={"padding": "1rem"})


# 3) Callback to update all visuals
@callback(
    Output("pca-cluster-scatter", "figure"),
    Output("pca-histograms",      "figure"),
    Output("cluster-profiles",     "figure"),
    Output("confusion-matrix",     "figure"),
    Input("n-clusters",           "value")
)
def update_cluster_views(k):
    # --- run k-means ---
    km, labels = fit_kmeans(X, n_clusters=k)
    labels_str = labels.astype(str)

    # --- 3D PCA projection ---
    pca = PCA(n_components=3, random_state=42)
    pcs = pca.fit_transform(X)
    df_pca = pd.DataFrame(
        pcs, columns=["PC1", "PC2", "PC3"], index=X.index
    )
    df_pca["Cluster"] = labels_str
    df_pca["Cancer Status"] = y.astype(str)
    fig_pca3d = px.scatter_3d(
        df_pca, x="PC1", y="PC2", z="PC3",
        color="Cluster", symbol="Cancer Status",
        title=f"3D PCA Projection (k={k})"
    )

    # --- PCA component histograms ---
    df_melt = df_pca.melt(
        id_vars=["Cluster"],
        value_vars=["PC1", "PC2", "PC3"],
        var_name="Component",
        value_name="Value"
    )
    fig_hist = px.histogram(
        df_melt,
        x="Value",
        color="Cluster",
        facet_col="Component",
        title="PCA Component Distributions",
        labels={"Value": "Component Value"},
        barmode="overlay"
    )

    # --- Cluster feature‐mean profiles ---
    df_prof = X.copy()
    df_prof["Cluster"] = labels_str
    means = df_prof.groupby("Cluster").mean().reset_index()
    df_melt2 = means.melt(
        id_vars="Cluster", var_name="Feature", value_name="Mean"
    )
    fig_prof = px.bar(
        df_melt2,
        x="Feature", y="Mean",
        facet_col="Cluster",
        title=f"Cluster Feature Means (k={k})",
        labels={"Mean": "Average (scaled)", "Feature": ""}
    )
    fig_prof.update_layout(showlegend=False)
    fig_prof.update_xaxes(tickangle=45)

    # --- Confusion matrix of cluster vs true label (only for k=2) ---
    if k == 2:
        cm = confusion_matrix(y, labels)
        fig_cm = px.imshow(
            cm,
            x=[f"Pred {c}" for c in sorted(set(labels))],
            y=[f"True {c}" for c in sorted(set(y))],
            text_auto=True,
            title="Confusion Matrix: Cluster vs Cancer Status"
        )
    else:
        fig_cm = go.Figure()
        fig_cm.add_annotation(
            text="Confusion matrix only available for k=2",
            x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False
        )
        fig_cm.update_xaxes(visible=False)
        fig_cm.update_yaxes(visible=False)

    return fig_pca3d, fig_hist, fig_prof, fig_cm
