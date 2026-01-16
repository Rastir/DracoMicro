# Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY rf_optimized.joblib ./rf_optimized.joblib
COPY model_columns.joblib ./model_columns.joblib
COPY clientes_limpio.csv ./clientes_limpio.csv
COPY rf_best.joblib ./rf_best.joblib

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
