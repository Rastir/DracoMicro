from joblib import load
from app.schemas import ChurnInput


class ModelService:
    def __init__(self, model_path="rf_best.joblib"):
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

model_service = ModelService()
