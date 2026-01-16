import pandas as pd

def preprocess_input(payload: dict) ->  pd.DataFrame:
    payload = dict(payload)
    df = pd.DataFrame([payload])  # Convert dict → DataFrame
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df = pd.get_dummies(df)
    return df

def lowercase_dict_values(d: dict) -> dict:
    return {
        k: (v.lower().strip() if isinstance(v, str) else v)
        for k, v in d.items()
    }

def lowercase_df(df):
    df = df.copy()
    for col in df.select_dtypes(include=["object", "string", "category"]).columns:
        df[col] = df[col].str.lower().str.strip()
    return df
