# Classification page

# libraries/imports 
import dash
from dash import html, dcc, callback, Output, Input
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from sklearn.metrics import confusion_matrix
from utils.data_loader import load_raw_data, preprocess_df, get_train_test
from utils.model_utils import train_classifiers, get_metrics, get_roc_curve

# registering page for directory
dash.register_page(__name__, path="/classification")

# loading data 
raw_df = load_raw_data()

# processing data 
X, y = preprocess_df(raw_df)

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

# gettign training and testing split
X_train, X_test, y_train, y_test = get_train_test(X, y)

# building models (once in the back end) 
models = train_classifiers(X_train, y_train)

# getting model metrics
metrics_df = get_metrics(models, X_test, y_test)

# page layout
layout = html.Div([
    # main header/summary 
    html.H2("Classification Performance"),
    html.P(
        "Classification analysis involved 3 different modelling approaches: Logistic Regression, Extreme Gradient Boosting (XGBoost), " 
        "and Random Forest. The results of each of these approaches are detailed below. Generally, all 3 classification models " 
        "showed similar, poor, performance. As seen by the similar accuracies and ROC curves, the " 
        "performance of classification models does not differ greatly between the different approaches. "
    ),

    # dropdown menu for each model
     html.P(
        "Choose a modelling approach to examine. " 
    ),
    dcc.Dropdown(
        id="model-select",
        options=[{"label": name, "value": name} for name in models.keys()],
        value=list(models.keys())[0],
        clearable=False
    ),

    # model metrics & accuracy 
    html.P(
        "When optimized, this modelling approach has the followng accuracy and respective hyperparameters: "
    ),
    html.Div(id="metrics-container", style={"marginTop": "1rem"}),
    dcc.Graph(id="roc-curve", style={"marginTop": "1rem"}),

    # confusion matrix to see overall predictice power 
    html.H3("Predictive Accuracy of Model"),
    html.P(
        "The accuracy and predictive power of model is shown below with a confusion matrix."
    ),
    dcc.Graph(id="confusion-matrix", style={"marginTop": "1rem"}),

    # feature deep dive for random forest and XGB
    html.H3("Feature Insights for Classification Models"),
    html.P(
        "A secondary analysis was conducted to examine the relative importance of each feature within the dataset for cancer classification. " 
        "These variables which appear to have the strongest influence on classification may highlight areas for further analysis. "
        ),
    dcc.Graph(id="feature-plot", style={"marginTop": "1rem"}),
])

# callback fopr updating metrics 
@callback(
    Output("metrics-container", "children"),
    Input("model-select", "value")
)
# function for metric update
def update_metrics_display(model_name):
    # getting metrics for specific model
    row = metrics_df.loc[model_name]
    # formatting names
    display_names = {
        "accuracy": "Accuracy",
        "precision": "Precision",
        "recall": "Recall",
        "AUC": "AUC"
    }
    # getting accuracy + rounding
    accuracy_line = html.P(
        f"Accuracy: {row['accuracy']:.3f}",
        style={"fontWeight": "bold"}
    )
    # forming table for all metrics
    table_header = html.Thead(html.Tr([html.Th("Metric"), html.Th("Value")]))
    table_rows = []
    for metric, value in row.items():
        name = display_names.get(metric, metric)
        table_rows.append(html.Tr([html.Td(name), html.Td(f"{value:.3f}")]))
    table_body = html.Tbody(table_rows)
    table = html.Table([table_header, table_body], style={"borderCollapse": "collapse", "width": "50%"})
    return [accuracy_line, table]

# callback for ROC curve update
@callback(
    Output("roc-curve", "figure"),
    Input("model-select", "value")
)
# function for plotting ROC update
def plot_roc_curve(model_name):
    fpr, tpr, auc_val = get_roc_curve(models[model_name], X_test, y_test)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC = {auc_val:.3f}"))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
    fig.update_layout(title=f"ROC Curve: {model_name}", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return fig

# callback for feature plot update
@callback(
    Output("feature-plot", "figure"),
    Input("model-select", "value")
)
# function for mapping feature importance for XGB OR random forest feature (# of splits)
def update_feature_plot(model_name):
    # Plot for random forest
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

    # Plot for XGB AKA number of splits per feature ('weight' importance)
    elif model_name == "XGBoost":
        booster = models[model_name].get_booster()
        importance_dict = booster.get_score(importance_type='weight')
        # sorting by importance 
        features = list(importance_dict.keys())
        scores = list(importance_dict.values())
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        features = [features[i] for i in sorted_idx]
        scores = [scores[i] for i in sorted_idx]

        # deining colors from Plotly pastel palette 
        colors = (px.colors.qualitative.Pastel * ((len(features) // len(px.colors.qualitative.Pastel)) + 1))[:len(features)]

        # building df for plotting
        imp_df = pd.DataFrame({"Feature": features, "NumSplits": scores})

        # bar chart for feature splits 
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=imp_df['NumSplits'][::-1],
            y=imp_df['Feature'][::-1],
            orientation='h',
            marker_color=colors[::-1],
            text=[f"{s:.1f}" for s in scores[::-1]],
            textposition='outside',
        ))
        # formatting XGB graph 
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

    # creating placeholder for logistic regression 
    fig = go.Figure()
    fig.add_annotation(text="Feature insights only available for RandomForest & XGBoost",
                       x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)

    return fig

# confusion matrix callback
@callback(
    Output("confusion-matrix", "figure"),
    Input("model-select", "value")
)
# function for confusion matrix 
def plot_confusion(model_name):
    model = models[model_name]
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    class_names = ["Cancer Absent (0)", "Cancer Present (1)"] # formatting names 
    # creating matrix 
    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale=px.colors.sequential.Plasma_r,
        labels={"x": "Predicted", "y": "Actual"},
        title=f"Confusion Matrix: {model_name}"
        )
    # formatting layout
    fig.update_layout(
         xaxis=dict(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=class_names
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=class_names,
            autorange="reversed"
        )
    )

    return fig
