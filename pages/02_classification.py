# pages/02_classification.py

import dash
from dash import html, dcc, callback, Output, Input
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.metrics import confusion_matrix
from utils.data_loader import load_raw_data, preprocess_df, get_train_test
from utils.model_utils import train_classifiers, get_metrics, get_roc_curve

# 1) Register this page
dash.register_page(__name__, path="/classification")

# 2) Prepare data & models once
raw_df = load_raw_data()
X, y = preprocess_df(raw_df)

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

X_train, X_test, y_train, y_test = get_train_test(X, y)
models = train_classifiers(X_train, y_train)
metrics_df = get_metrics(models, X_test, y_test)

# 3) Page layout: dropdown, metrics table, ROC curve, feature/importance plots
layout = html.Div([
    html.H2("Classification Performance"),
    dcc.Dropdown(
        id="model-select",
        options=[{"label": name, "value": name} for name in models.keys()],
        value=list(models.keys())[0],
        clearable=False
    ),
    html.Div(id="metrics-container", style={"marginTop": "1rem"}),
    dcc.Graph(id="roc-curve", style={"marginTop": "1rem"}),
    dcc.Graph(id="feature-plot", style={"marginTop": "1rem"}),
    dcc.Graph(id="confusion-matrix", style={"marginTop": "1rem"}),
])

# 4) Callback: update metrics table + accuracy line
@callback(
    Output("metrics-container", "children"),
    Input("model-select", "value")
)
def update_metrics_display(model_name):
    # Extract metrics for selected model
    row = metrics_df.loc[model_name]
    display_names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "AUC": "AUC"
    }
    # Accuracy line
    accuracy_line = html.P(
        f"Accuracy: {row['accuracy']:.3f}",
        style={"fontWeight": "bold"}
    )
    # Metrics table
    table_header = html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")]))
    table_rows = []
    for metric, value in row.items():
        name = display_names.get(metric, metric)
        table_rows.append(html.Tr([html.Td(name), html.Td(f"{value:.3f}")]))
    table_body = html.Tbody(table_rows)
    table = html.Table([table_header, table_body], style={"borderCollapse": "collapse", "width": "50%"})
    return [accuracy_line, table]

# 5) Callback: update ROC curve
@callback(
    Output("roc-curve", "figure"),
    Input("model-select", "value")
)
def plot_roc_curve(model_name):
    fpr, tpr, auc_val = get_roc_curve(models[model_name], X_test, y_test)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc_val:.3f}"))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    fig.update_layout(title=f"ROC Curve: {model_name}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig

# 6) Callback: update feature importance / split count plot
@callback(
    Output("feature-plot", "figure"),
    Input("model-select", "value")
)
def update_feature_plot(model_name):
    # Random Forest feature importances
    if model_name == "RandomForest":
        importances = models[model_name].feature_importances_
        imp_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": importances
        }).sort_values("Importance", ascending=False)
        fig = px.bar(
            imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            title="Feature Importances (Random Forest)",
            color="Feature",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
        return fig

    # XGBoost: number of splits per feature ('weight' importance)
    elif model_name == "XGBoost":
        booster = models[model_name].get_booster()
        importance_dict = booster.get_score(importance_type='weight')
        # Sort features by importance
        features = list(importance_dict.keys())
        scores = list(importance_dict.values())
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        features = [features[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]

        # Custom pastel palette (repeated)
        colors = (px.colors.qualitative.Pastel * ((len(features) // len(px.colors.qualitative.Pastel)) + 1))[:len(features)]

        # Build a DataFrame for ease
        imp_df = pd.DataFrame({"Feature": features, "NumSplits": scores})

        # Plot horizontal bar with custom colors
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=imp_df['NumSplits'][::-1],
            y=imp_df['Feature'][::-1],
            orientation='h',
            marker_color=colors[::-1],
            text=[f"{s:.1f}" for s in scores[::-1]],
            textposition='outside',
        ))
        fig.update_layout(
            title="XGBoost Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            yaxis=dict(categoryorder="array", categoryarray=features[::-1]),
            showlegend=False,
            plot_bgcolor='white'
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        return fig

    # Placeholder for other models
    fig = go.Figure()
    fig.add_annotation(text="Feature insights only available for RandomForest & XGBoost",
                       x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig

@callback(
    Output("confusion-matrix", "figure"),
    Input("model-select", "value")
)

def plot_confusion(model_name):
    model = models[model_name]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    labels = list(model.classes_)
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale=px.colors.sequential.Plasma_r,
        labels={"x": "Predicted", "y": "Actual"},
        title=f"Confusion Matrix: {model_name}"
        )
    fig.update_layout(yaxis=dict(autorange="reversed"))

    return fig
