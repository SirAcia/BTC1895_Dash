import dash
from dash import html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from utils.data_loader import load_raw_data, preprocess_df
from utils.model_utils import fit_kmeans

dash.register_page(__name__, path="/clustering")

df_raw = load_raw_data()
X, y   = preprocess_df(df_raw)

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

    html.H3("Cluster Feature Means (Scaled)"),
    dcc.Graph(id="cluster-profiles"),

    html.H3("Cluster vs. Cancer Status Confusion Matrix"),
    dcc.Graph(id="confusion-matrix-cluster"),

    html.H3("Predicted Class vs. True Cancer Status Confusion Matrix"),
    dcc.Graph(id="confusion-matrix-predicted"),
], style={"padding": "1rem"})

@callback(
    Output("pca-cluster-scatter", "figure"),
    Output("cluster-profiles",     "figure"),
    Output("confusion-matrix-cluster",   "figure"),
    Output("confusion-matrix-predicted", "figure"),
    Input("n-clusters",           "value")
)
def update_cluster_views(k):
    km, labels = fit_kmeans(X, n_clusters=k)
    labels_str = labels.astype(str)

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
        color="Feature",
        title=f"Cluster Feature Means (k={k})",
        labels={"Mean": "Average (scaled)", "Feature": ""}
    )
    fig_prof.update_layout(showlegend=False)
    fig_prof.update_xaxes(tickangle=45)

    cm = confusion_matrix(y, labels)
    fig_cm_cluster = px.imshow(
        cm,
        x=[f"Pred Cluster {c}" for c in sorted(np.unique(labels))],
        y=[f"True {c}" for c in sorted(np.unique(y))],
        text_auto=True,
        title="Confusion Matrix: Cluster vs Cancer Status"
    )


    cluster_to_class = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)
        true_labels = y.iloc[idx]
        cluster_to_class[c] = true_labels.mode().iat[0]

    y_pred = pd.Series(labels).map(cluster_to_class)

    cm_pred = confusion_matrix(y, y_pred)
    fig_cm_pred = px.imshow(
        cm_pred,
        x=[f"Pred Class {cls}" for cls in sorted(np.unique(y_pred))],
        y=[f"True {cls}" for cls in sorted(np.unique(y))],
        text_auto=True,
        title="Confusion Matrix: Predicted Class vs True Cancer Status"
    )

    return fig_pca3d, fig_prof, fig_cm_cluster, fig_cm_pred