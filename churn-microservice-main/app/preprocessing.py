import pandas as pd

def preprocess_input(payload: dict) ->  pd.DataFrame:
    payload = dict(payload)
    df = pd.DataFrame([payload])  # Convert dict → DataFrame
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df = pd.get_dummies(df)
    return df
