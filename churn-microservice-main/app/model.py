from joblib import load
import pandas as pd
from app.schemas import ChurnInput


class ModelService:
    def __init__(self, model_path="rf_optimized.joblib"):
        self.model = load(model_path)

    def predict(self, data: ChurnInput):
        df = data
        pred = self.model.predict(df)[0]
        prob = self.model.predict_proba(df)[0].tolist()

        return {
            "prediction": int(pred),
            "probabilities": {
                "not_churn": prob[0],
                "churn": prob[1]
            }
        }
    
    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        probs = self.model.predict_proba(df)
        df = df.copy()
        df["not_churn_prob"] = probs[:, 0]
        df["churn_prob"] = probs[:, 1]

        return df

model_service = ModelService()
