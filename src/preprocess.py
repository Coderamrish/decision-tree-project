import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(path):
    df = pd.read_csv(path)

    df2 = df.copy()
    le = LabelEncoder()

    for col in df2.columns:
        if df2[col].dtype == object:
            df2[col] = le.fit_transform(df2[col])

    X = df2.drop("status", axis=1)
    y = df2["status"]

    return X, y
