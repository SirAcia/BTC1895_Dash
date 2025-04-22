# data loader in utils page 

# libraries/imports 
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# function for reading data (so only read once)
def load_raw_data(path="data/raw/synthetic_cancer_data.csv"):
    df = pd.read_csv(path, engine="python")
    # renaming variables 
    cat_cols = [
        "Sex","Smoking_Status","Family_History",
        "TP53_Mutation","BRCA1_Mutation","KRAS_Mutation",
        "Tumor_Location","Cancer_Status"
    ]
    # converting variables to categorical
    df[cat_cols] = df[cat_cols].astype("category")
    return df

# function for processing data (remove negative + scaling)
def preprocess_df(df):
    # removing patient ID + cancer status for X
    X = df.iloc[:, 1:-1]

    y = df["Cancer_Status"]

    mask = y.isin([0,1])

    X = X.loc[mask]
    y = y.loc[mask]

    # imputing numeric columns
    num_cols = X.select_dtypes(include=[float, int]).columns
    imputer = SimpleImputer(strategy="median")
    X[num_cols] = imputer.fit_transform(X[num_cols])

    # scaling using standard scaler 
    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    cat_cols = X.select_dtypes(include="category").columns
    for c in cat_cols:
        X[c] = X[c].cat.codes

    return X, y

# function for splitting data into train + test 
def get_train_test(X, y, test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
