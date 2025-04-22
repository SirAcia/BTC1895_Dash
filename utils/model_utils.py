# model building in utils page 

# libraries/imports 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve, auc
import pandas as pd

# training classifiers 
def train_classifiers(X_train, y_train):
    models = {}

    # training logistic regression model 
    lr = LogisticRegression(max_iter=1_000)
    models["Logistic"] = lr.fit(X_train, y_train)

   # training random forest model 
    rf = RandomForestClassifier(random_state=42)
    models["RandomForest"] = rf.fit(X_train, y_train)

    # training XGBoost
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    models["XGBoost"] = xgb_clf.fit(X_train, y_train)

    return models

# function for pulling metrics for classification models
def get_metrics(models, X_test, y_test):
    records = []
    for name, m in models.items():
        y_pred = m.predict(X_test)
        y_proba = m.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        records.append({
            "model": name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "AUC": auc(fpr, tpr)
        })
    return pd.DataFrame.from_records(records).set_index("model")

# function for plotting ROC curves
def get_roc_curve(model, X_test, y_test):
    y_proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return fpr, tpr, auc(fpr, tpr)

# function for clustering with K Means
def fit_kmeans(X, n_clusters=2):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X_scaled)
    return km, labels
