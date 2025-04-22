# home page dash 

# libraries
import dash
from dash import html

# registering page for directory
dash.register_page(__name__, path="/")

# page layout 
layout = html.Div([
    # main header
    html.H2("Summary of Analysis & Findings"),
    html.P(
        "This interactive dashboard explores the use of machine learning (ML) models to classifying/clustering patients " 
        "based on a synthetic cancer dataset of 1,000 individuals. The dataset includes a range of clinical & demographic data, " 
        "genetic data, biomarker & blood test data, and imaging features relevant for cancer diagnosis."),

    # secondary header 
    html.H3("Project Aims & Workflow"), 

    html.P(
        "The aim of the project was two-fold, identifying supervised modelling approaches to predict cancer status in patients as well as "
        "unsupervised approaches to understand prominent features through clustering techniques. Data cleaning involved the removal of negative values, "
        "and categorization of variables. Imputation and scaling varied between modelling approaches. "),

    # secondary header
    html.H3("Results & Next Steps"), 

    html.P(
        "Three classification models - Logistic, Random Forest, Extreme Gradient Boost (XGBoost) - were trained to detect the presence of cancer,"
        "while a K-Means clustering model was employed to explore any potential naturally-occuring groupings within the data."),
    html.P(
        "All 3 classification models showed similar levels of poor performance, being unable to effectively separate cancer from non-cancer cases on test data, "
        "with models showing high levels of false positives and negatives. Performance was slightly better for the positive class (cancer positive)"
        "however, when considering class imbalances, the models had poor performance across both classes." 
    ),
    html.P(
        "Similarly, cluster analysis showed no statistically significant grouping based on cancer status as well as no distinct differences in feature compositon between clusters. "
    ),
    html.P(
        "Overall, these results underline the importance of richer and more granular data for predictive oncology modeling. With greater granularity and data quality, "
        "identifcation of factors accounting for unexplained variance can be identified which can further be used to identify key features and/or cluster demographics, "
        "impacting future research directions (i.e. targetting of specific genes for specific cancers) and business strategies"
    )
    ])