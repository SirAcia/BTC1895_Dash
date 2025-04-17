import dash
from dash import html, dcc
import plotly.express as px
from utils.data_loader import load_raw_data, preprocess_df
from utils.model_utils import fit_kmeans

dash.register_page(__name__, path="/clustering")

# prepare data once
df_raw = load_raw_data()
X, y = preprocess_df(df_raw)
df_clust = X.copy()
df_clust["Cancer_Status"] = y

layout = html.Div([
    html.H2("K‑Means Clustering"),
    dcc.Slider(id="n-clusters", min=2, max=10, step=1, value=2),
    dcc.Graph(id="cluster-scatter")
])

@dash.callback(
    dash.Output("cluster-scatter", "figure"),
    dash.Input("n-clusters", "value")
)
def update_clusters(k):
    km, labels = fit_kmeans(X, n_clusters=k)
    df_plot = df_clust.copy()
    df_plot["Cluster"] = labels
    fig = px.scatter(df_plot, x=X.columns[0], y=X.columns[1],
                     color="Cluster", symbol="Cancer_Status",
                     title=f"KMeans (k={k})")
    return fig
