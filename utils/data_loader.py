# utils/data_loader.py
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_raw_data(path="data/synthetic_cancer_data.csv"):
    df = pd.read_csv(path, engine="python")
    # cast the categorical columns exactly as in your NB
    cat_cols = [
        "Sex","Smoking_Status","Family_History",
        "TP53_Mutation","BRCA1_Mutation","KRAS_Mutation",
        "Tumor_Location","Cancer_Status"
    ]
    df[cat_cols] = df[cat_cols].astype("category")
    return df

def preprocess_df(df):
    # 1) separate X / y exactly as in your NB
    X = df.iloc[:, 1:-1]
    y = df["Cancer_Status"]
    mask = y.isin([0,1])
    X = X.loc[mask]
    y = y.loc[mask]

    # 2) impute numeric columns
    num_cols = X.select_dtypes(include=[float, int]).columns
    imputer = SimpleImputer(strategy="median")
    X[num_cols] = imputer.fit_transform(X[num_cols])

    # 3) scale numeric
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    # 4) encode categories as codes
    cat_cols = X.select_dtypes(include="category").columns
    for c in cat_cols:
        X[c] = X[c].cat.codes

    return X, y

def get_train_test(X, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
