from app.preprocessing import lowercase_df, lowercase_dict_values
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import ChurnInput
from app.model import model_service
import pandas as pd

app = FastAPI(
    title="Netflix Churn Prediction API",
    version="1.0.0"
)

# Configuración CORS para producción
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
        data = lowercase_dict_values(data.dict())
        df = pd.DataFrame([data])
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

    df_pred = result[["age", "gender", "subscription_type", "watch_hours", "region","number_of_profiles", "payment_method", "device"]]
    df_pred = lowercase_df(df_pred)
    result_pred = model_service.predict(df_pred)
    return {"status": "success", 'data': result.to_dict(orient='records'), 'prediction': result_pred}


@app.get("/probability/age")
def get_probability_by_age():
    result = load_csv()
    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Información no fue encontrada"
        )

    df_pred = result[["age", "gender", "subscription_type", "watch_hours", "region", "number_of_profiles", "payment_method", "device"]]
    df_pred = lowercase_df(df_pred)
    df_pred = model_service.predict_batch(df_pred)
    grouped = (
        df_pred
        .groupby("age")
        .agg(
            churn_probability=("churn_prob", "mean"),
            not_churn_probability=("not_churn_prob", "mean"),
            users_count=("age", "count")
        )
        .reset_index()
    )

    # Convertir en porcentajes
    grouped["churn_probability"] = (grouped["churn_probability"] * 100).round(2)
    grouped["not_churn_probability"] = (grouped["not_churn_probability"] * 100).round(2)

    # return JSON
    return {
        "total_users": len(df_pred),
        "grouped_by_age": grouped.to_dict(orient="records")
    }

@app.get("/probability/gender")
def get_probability_by_gender():
    result = load_csv()
    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Información no fue encontrada"
        )

    df_pred = result[["age", "gender", "subscription_type", "watch_hours", "region", "number_of_profiles", "payment_method", "device"]]
    df_pred = lowercase_df(df_pred)
    df_pred = model_service.predict_batch(df_pred)
    grouped = (
        df_pred
        .groupby("gender")
        .agg(
            churn_probability=("churn_prob", "mean"),
            not_churn_probability=("not_churn_prob", "mean"),
            users_count=("gender", "count")
        )
        .reset_index()
    )

    # Convertir en porcentajes
    grouped["churn_probability"] = (grouped["churn_probability"] * 100).round(2)
    grouped["not_churn_probability"] = (grouped["not_churn_probability"] * 100).round(2)

    # return JSON
    return {
        "total_users": len(df_pred),
        "grouped_by_gender": grouped.to_dict(orient="records")
    }

@app.get("/probability/subscription")
def get_probability_by_subscription_typer():
    result = load_csv()
    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Información no fue encontrada"
        )

    df_pred = result[["age", "gender", "subscription_type", "watch_hours", "region", "number_of_profiles", "payment_method", "device"]]
    df_pred = lowercase_df(df_pred)
    df_pred = model_service.predict_batch(df_pred)
    grouped = (
        df_pred
        .groupby("subscription_type")
        .agg(
            churn_probability=("churn_prob", "mean"),
            not_churn_probability=("not_churn_prob", "mean"),
            users_count=("subscription_type", "count")
        )
        .reset_index()
    )

    # Convertir en porcentajes
    grouped["churn_probability"] = (grouped["churn_probability"] * 100).round(2)
    grouped["not_churn_probability"] = (grouped["not_churn_probability"] * 100).round(2)

    # return JSON
    return {
        "total_users": len(df_pred),
        "grouped_by_subscription_type": grouped.to_dict(orient="records")
    }

@app.get("/probability/region")
def get_probability_by_region():
    result = load_csv()
    if result.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Información no fue encontrada"
        )

    df_pred = result[["age", "gender", "subscription_type", "watch_hours", "region", "number_of_profiles", "payment_method", "device"]]
    df_pred = lowercase_df(df_pred)
    df_pred = model_service.predict_batch(df_pred)
    grouped = (
        df_pred
        .groupby("region")
        .agg(
            churn_probability=("churn_prob", "mean"),
            not_churn_probability=("not_churn_prob", "mean"),
            users_count=("region", "count")
        )
        .reset_index()
    )

    # Convertir en porcentajes
    grouped["churn_probability"] = (grouped["churn_probability"] * 100).round(2)
    grouped["not_churn_probability"] = (grouped["not_churn_probability"] * 100).round(2)

    # return JSON
    return {
        "total_users": len(df_pred),
        "grouped_by_region": grouped.to_dict(orient="records")
    }