from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ChurnInput
from app.model import model_service
import pandas as pd

app = FastAPI(
    title="Netflix Churn Prediction API",
    version="1.0.0"
)

# Configuración CORS
origins = [
    "https://dracostack.com",
    "http://dracostack.com",
    "https://www.dracostack.com",
    "http://www.dracostack.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
)

CSV_PATH = "clientes_limpio.csv"

def load_csv():
    return pd.read_csv(CSV_PATH)

@app.get("/item/{item_id}")
def get_item(item_id: str):
    df = load_csv()

    # Filtrando por public id (string comparison)
    result = df[df["public_id"] == item_id]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Información con id '{item_id}' no fue encontrado"
        )

    # Convertir csv a JSON-dict
    row_dict = result.to_dict(orient="records")[0]

    return {"status": "success", "data": row_dict}

@app.get("/items")
def get_all_items():
    df = load_csv()
    return {
        "status": "success",
        "total": len(df),
        "data": df.to_dict(orient="records")
    }

@app.post("/predict")
def predict(data: ChurnInput):
    try:
        df = pd.DataFrame([data.dict()])
        result = model_service.predict(df)

        return result

    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@app.get("/item/predictions/{item_id}")
def get_items_predictions(item_id: str):
    df = load_csv()

    # Filtrando por public id (string comparison)
    result = df[df["public_id"] == item_id]

    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Información con id '{item_id}' no fue encontrado"
        )

    df_pred = result[['age', 'gender', 'subscription_type', 'watch_hours', 'last_login_days', 'region']]
    result_pred = model_service.predict(df_pred)
    return {"status": "success", 'data': result.to_dict(orient='records'), 'prediction': result_pred}