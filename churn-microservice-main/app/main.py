from app.preprocessing import lowercase_df
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.schemas import ChurnInput
from app.model import model_service
import pandas as pd
import traceback

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

# Handler para errores de validación (422)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Captura errores de validación y muestra información detallada
    para facilitar el debugging
    """
    print("❌ ERROR DE VALIDACIÓN:")
    print(f"Errores: {exc.errors()}")
    print(f"Body recibido: {exc.body}")

    return JSONResponse(
        status_code=422,
        content={
            "detail": exc.errors(),
            "message": "Los datos enviados no cumplen con el formato esperado",
            "body_received": exc.body
        }
    )

CSV_PATH = "clientes_limpio.csv"

def load_csv():
    return pd.read_csv(CSV_PATH)

@app.get("/")
def root():
    """Endpoint raíz para verificar que la API está funcionando"""
    return {
        "message": "Netflix Churn Prediction API",
        "version": "1.0.0",
        "status": "online"
    }

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
    """
    Predice si un cliente va a abandonar el servicio (churn).
    Los datos ya vienen validados y en minúsculas gracias al schema.
    """
    try:
        print("📥 Datos recibidos y validados:", data.dict())

        # Ya NO necesitamos lowercase_dict_values()
        # porque el validator del schema ya lo hace
        df = pd.DataFrame([data.dict()])
        print("✅ DataFrame creado exitosamente")

        result = model_service.predict(df)
        print("✅ Predicción realizada:", result)

        return result

    except Exception as exc:
        print("❌ ERROR EN PREDICCIÓN:")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error al realizar la predicción: {str(exc)}"
        )


@app.post("/predict/debug")
async def predict_debug(request: Request):
    """
    Endpoint temporal para debugging.
    Muestra exactamente qué datos están llegando al servidor.
    """
    try:
        body = await request.json()
        print("📦 Datos RAW recibidos:", body)
        print("📋 Tipos de datos:", {k: type(v).__name__ for k, v in body.items()})

        return {
            "status": "success",
            "message": "Datos recibidos correctamente",
            "data_received": body,
            "data_types": {k: type(v).__name__ for k, v in body.items()}
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


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

    return {
        "status": "success",
        "data": result.to_dict(orient='records'),
        "prediction": result_pred
    }


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