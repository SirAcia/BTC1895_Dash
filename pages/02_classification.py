import dash
from dash import html, dcc
import plotly.graph_objects as go
from utils.data_loader import load_raw_data, preprocess_df, get_train_test
from utils.model_utils import train_classifiers, get_metrics, get_roc_curve

dash.register_page(__name__, path="/classification")

# Prepare data & models once
df_raw = load_raw_data()
X, y = preprocess_df(df_raw)
X_train, X_test, y_train, y_test = get_train_test(X, y)
models = train_classifiers(X_train, y_train)
metrics_df = get_metrics(models, X_test, y_test)

layout = html.Div([
    html.H2("Classification Performance"),
    dcc.Dropdown(
        id="model-select",
        options=[{"label": m, "value": m} for m in models],
        value="Logistic"
    ),
    dcc.Graph(id="metrics-bar"),
    dcc.Graph(id="roc-curve")
])

@dash.callback(
    dash.Output("metrics-bar", "figure"),
    dash.Input("model-select", "value")
)
def plot_metrics(model_name):
    row = metrics_df.loc[model_name]
    fig = go.Figure([go.Bar(x=row.index, y=row.values)])
    fig.update_layout(title=f"{model_name} Metrics")
    return fig

@dash.callback(
    dash.Output("roc-curve", "figure"),
    dash.Input("model-select", "value")
)
def plot_roc(model_name):
    fpr, tpr, auc_val = get_roc_curve(models[model_name], X_test, y_test)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"AUC={auc_val:.3f}"))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(dash="dash"))
    fig.update_layout(title=f"ROC Curve: {model_name}",
                      xaxis_title="FPR", yaxis_title="TPR")
    return fig
