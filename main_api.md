# API de previsión de demanda — `main.py`

API en Python (FastAPI) del Trabajo Fin de Grado *"Diseño de un sistema de
previsión de demanda y planificación de la producción mediante flujos
automatizados con IA"*. Expone los endpoints que n8n llama para seleccionar
el modelo de previsión, generar el forecast y evaluar su error frente a las
ventas reales.

## Stack

- **FastAPI** — servidor y definición de endpoints.
- **SQLAlchemy** sobre **PostgreSQL** — almacenamiento de ventas, modelos,
  previsiones y errores.
- **scikit-learn**, **XGBoost**, **statsmodels (SARIMAX)** y, si está
  disponible, **Prophet** — modelos de previsión candidatos.

## Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/health` | Comprobación de estado del servicio. |
| POST | `/select_model` | Backtesting de los modelos candidatos y guarda el mejor para el SKU. |
| POST | `/retrain` | Reentrena y vuelve a seleccionar el mejor modelo (últimos 24 meses). |
| POST | `/forecast` | Genera la previsión con el modelo ya seleccionado. |
| POST | `/evaluate` | Compara previsiones pasadas con las ventas reales y registra el error. |
| GET | `/models/{sku}` | Información del modelo seleccionado para un SKU. |
| GET | `/errors/{sku}` | Últimos errores de previsión registrados para un SKU. |

## Código completo

```python
import os
import json
import math
import joblib
import warnings
from datetime import date
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except Exception:
    PROPHET_AVAILABLE = False


# =====================================================
# CONFIGURACIÓN
# =====================================================

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://forecast_user:TU_PASSWORD@forecasting-db:5432/forecasting"
)

MODEL_DIR = "saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

DEFAULT_HORIZON = 3
TRAINING_WINDOW_MONTHS = int(os.getenv("TRAINING_WINDOW_MONTHS", "24"))

engine = create_engine(DATABASE_URL)

app = FastAPI(title="Demand Forecasting API", version="1.1.0")


# =====================================================
# MODELOS DE REQUEST
# =====================================================

class SKURequest(BaseModel):
    sku: str


class FutureFeature(BaseModel):
    fecha: date
    pedidos_comprometidos: Optional[float] = 0
    precio: Optional[float] = 0
    promo: Optional[int] = 0
    evento: Optional[str] = "ninguno"
    intensidad_evento: Optional[float] = 0


class ForecastRequest(BaseModel):
    sku: str
    horizon: int = DEFAULT_HORIZON
    save_forecast: bool = True
    future_features: Optional[List[FutureFeature]] = None


class EvaluateRequest(BaseModel):
    sku: str


# =====================================================
# UTILIDADES BD
# =====================================================

def read_sql(query: str, params: Optional[dict] = None) -> pd.DataFrame:
    return pd.read_sql(query, engine, params=params or {})


def execute_sql(query: str, params: Optional[dict] = None):
    with engine.begin() as conn:
        conn.execute(text(query), params or {})


def init_tables():
    execute_sql("""
        CREATE TABLE IF NOT EXISTS ventas (
            id SERIAL PRIMARY KEY,
            sku TEXT NOT NULL,
            fecha DATE NOT NULL,
            ventas NUMERIC NOT NULL,
            pedidos_comprometidos NUMERIC,
            precio NUMERIC,
            promo INTEGER,
            evento TEXT,
            intensidad_evento NUMERIC,
            created_at TIMESTAMP DEFAULT NOW(),
	    UNIQUE (sku, fecha)
        );
    """)

    execute_sql("""
        CREATE TABLE IF NOT EXISTS modelos (
            sku TEXT PRIMARY KEY,
            modelo TEXT,
            ruta_modelo TEXT,
            mape NUMERIC,
            mae NUMERIC,
            rmse NUMERIC,
            metricas JSONB,
            ultima_actualizacion TIMESTAMP DEFAULT NOW()
        );
    """)

    execute_sql("""
        CREATE TABLE IF NOT EXISTS errores (
            id SERIAL PRIMARY KEY,
            sku TEXT,
            fecha DATE,
            forecast NUMERIC,
            real NUMERIC,
            error_abs NUMERIC,
            error_pct NUMERIC,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    execute_sql("""
        CREATE TABLE IF NOT EXISTS forecasts (
            id SERIAL PRIMARY KEY,
            sku TEXT NOT NULL,
            fecha_forecast DATE NOT NULL,
            fecha_predicha DATE NOT NULL,
            forecast NUMERIC NOT NULL,
            modelo_usado TEXT,
            created_at TIMESTAMP DEFAULT NOW()
        );
    """)

    execute_sql("""
        CREATE TABLE IF NOT EXISTS variables_futuras (
            id SERIAL PRIMARY KEY,
            sku TEXT NOT NULL,
            fecha DATE NOT NULL,
            pedidos_comprometidos NUMERIC,
            precio NUMERIC,
            promo INTEGER,
            evento TEXT,
            intensidad_evento NUMERIC,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP DEFAULT NOW(),
            UNIQUE (sku, fecha)
        );
    """)

    execute_sql("""
        CREATE TABLE IF NOT EXISTS workflow_control (
            key TEXT PRIMARY KEY,
            value INTEGER NOT NULL DEFAULT 0,
            updated_at TIMESTAMP DEFAULT NOW()
        );
    """)

    # Índice requerido por el ON CONFLICT de save_forecasts. Sin él, guardar
    # previsiones falla en un despliegue nuevo hasta ejecutarlo desde n8n.
    execute_sql("""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_forecasts_sku_fecha_run
        ON forecasts (sku, fecha_forecast, fecha_predicha);
    """)

    execute_sql("""
        CREATE INDEX IF NOT EXISTS idx_forecasts_sku_fecha
        ON forecasts (sku, fecha_predicha);
    """)


@app.on_event("startup")
def startup_event():
    init_tables()


# =====================================================
# MÉTRICAS
# =====================================================

def safe_mape(y_true, y_pred) -> float:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    mask = y_true != 0

    if mask.sum() == 0:
        return 999.0

    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def calculate_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    return {
        "mape": safe_mape(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred)))
    }


# =====================================================
# CARGA Y PREPARACIÓN DE DATOS
# =====================================================

def get_sku_data(sku: str) -> pd.DataFrame:
    query = """
        SELECT *
        FROM ventas
        WHERE sku = %(sku)s
        AND fecha > (
            SELECT MAX(fecha) - (%(training_window_months)s * INTERVAL '1 month')
            FROM ventas
            WHERE sku = %(sku)s
        )
        ORDER BY fecha ASC;
    """

    df = read_sql(query, {
        "sku": sku,
        "training_window_months": TRAINING_WINDOW_MONTHS
    })

    if df.empty:
        raise HTTPException(status_code=404, detail=f"No hay datos para SKU {sku}")

    df["fecha"] = pd.to_datetime(df["fecha"])
    df["ventas"] = pd.to_numeric(df["ventas"], errors="coerce")

    numeric_cols = [
        "pedidos_comprometidos",
        "precio",
        "promo",
        "intensidad_evento"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "evento" in df.columns:
        df["evento"] = df["evento"].fillna("ninguno").astype(str)
    else:
        df["evento"] = "ninguno"

    df = df.dropna(subset=["fecha", "ventas"])

    return df


def encode_evento(df: pd.DataFrame):
    df = df.copy()
    encoder = LabelEncoder()
    df["evento_encoded"] = encoder.fit_transform(df["evento"].fillna("ninguno").astype(str))
    return df, encoder


def prepare_ml_features(
    df: pd.DataFrame,
    encoder: Optional[LabelEncoder] = None,
    dropna_target: bool = True
):
    df = df.copy()
    df = df.sort_values("fecha")

    df["mes"] = df["fecha"].dt.month
    df["trimestre"] = df["fecha"].dt.quarter

    df["lag_1"] = df["ventas"].shift(1)
    df["lag_2"] = df["ventas"].shift(2)
    df["lag_3"] = df["ventas"].shift(3)

    # Media móvil de los tres meses ANTERIORES: no incluye el mes en curso,
    # de modo que la variable puede calcularse también para un periodo futuro
    # y no incorpora información de la variable objetivo.
    df["media_3"] = df["ventas"].shift(1).rolling(3).mean()

    if encoder is None:
        df, encoder = encode_evento(df)
    else:
        known_classes = set(encoder.classes_)
        df["evento"] = df["evento"].apply(lambda x: x if x in known_classes else "ninguno")
        df["evento_encoded"] = encoder.transform(df["evento"].fillna("ninguno").astype(str))

    feature_cols = [
        "pedidos_comprometidos",
        "precio",
        "promo",
        "intensidad_evento",
        "evento_encoded",
        "mes",
        "trimestre",
        "lag_1",
        "lag_2",
        "lag_3",
        "media_3"
    ]

    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    df[feature_cols] = df[feature_cols].fillna(0)

    if dropna_target:
        df = df.dropna(subset=["ventas"])

    return df, feature_cols, encoder


def get_exog_columns(df: pd.DataFrame) -> List[str]:
    cols = [
        "pedidos_comprometidos",
        "precio",
        "promo",
        "intensidad_evento"
    ]

    existing = [c for c in cols if c in df.columns]

    for c in existing:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    return existing


# =====================================================
# MODELO 1: SEASONAL NAIVE
# =====================================================

def backtest_seasonal_naive(train: pd.DataFrame, test: pd.DataFrame):
    train_values = train["ventas"].values

    preds = []

    for i in range(len(test)):
        if len(train_values) >= 12:
            pred = train_values[-12]
        else:
            pred = train_values[-1]

        preds.append(pred)
        train_values = np.append(train_values, test["ventas"].iloc[i])

    metrics = calculate_metrics(test["ventas"].values, preds)

    model_info = {
        "model_name": "seasonal_naive",
        "last_values": train["ventas"].tail(12).tolist()
    }

    return preds, metrics, model_info


def forecast_seasonal_naive(model_info: dict, horizon: int):
    last_values = model_info["last_values"]

    if not last_values:
        return [0] * horizon

    preds = []

    for i in range(horizon):
        if len(last_values) >= 12:
            pred = last_values[i % 12]
        else:
            pred = last_values[-1]

        preds.append(float(pred))

    return preds


# =====================================================
# MODELO 2: SARIMAX
# =====================================================

def backtest_sarimax(train: pd.DataFrame, test: pd.DataFrame):
    try:
        train = train.copy()
        test = test.copy()

        exog_cols = get_exog_columns(train)

        y_train = train["ventas"].astype(float)

        if exog_cols:
            exog_train = train[exog_cols].astype(float)
            exog_test = test[exog_cols].astype(float).fillna(0)
        else:
            exog_train = None
            exog_test = None

        seasonal_period = 12 if len(train) >= 18 else 1

        model = SARIMAX(
            y_train,
            exog=exog_train,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, seasonal_period) if seasonal_period > 1 else (0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted = model.fit(disp=False)

        preds = fitted.forecast(
            steps=len(test),
            exog=exog_test if exog_cols else None
        )

        metrics = calculate_metrics(test["ventas"].values, preds.values)

        model_info = {
            "model_name": "sarimax",
            "model": fitted,
            "exog_cols": exog_cols
        }

        return preds.tolist(), metrics, model_info

    except Exception as e:
        return None, {"mape": 999, "mae": 999, "rmse": 999}, {"error": str(e)}


def forecast_sarimax(model_info: dict, df: pd.DataFrame, horizon: int, future_df: Optional[pd.DataFrame] = None):
    model = model_info["model"]
    exog_cols = model_info.get("exog_cols", [])

    if exog_cols:
        if future_df is not None:
            future_exog = future_df[exog_cols].astype(float).fillna(0)
        else:
            last_exog = df[exog_cols].tail(1).astype(float).fillna(0)
            future_exog = pd.concat([last_exog] * horizon, ignore_index=True)
    else:
        future_exog = None

    preds = model.forecast(
        steps=horizon,
        exog=future_exog if exog_cols else None
    )

    return [float(x) for x in preds]


# =====================================================
# MODELO 3: PROPHET
# =====================================================

def backtest_prophet(train: pd.DataFrame, test: pd.DataFrame):
    if not PROPHET_AVAILABLE:
        return None, {"mape": 999, "mae": 999, "rmse": 999}, {"error": "Prophet no disponible"}

    try:
        train_p = train.rename(columns={"fecha": "ds", "ventas": "y"}).copy()
        test_p = test.rename(columns={"fecha": "ds", "ventas": "y"}).copy()

        regressors = [
            "pedidos_comprometidos",
            "precio",
            "promo",
            "intensidad_evento"
        ]

        regressors = [r for r in regressors if r in train_p.columns]

        for r in regressors:
            train_p[r] = pd.to_numeric(train_p[r], errors="coerce").fillna(0)
            test_p[r] = pd.to_numeric(test_p[r], errors="coerce").fillna(0)

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )

        for r in regressors:
            model.add_regressor(r)

        model.fit(train_p[["ds", "y"] + regressors])

        future = test_p[["ds"] + regressors]

        forecast = model.predict(future)

        preds = forecast["yhat"].values

        metrics = calculate_metrics(test["ventas"].values, preds)

        model_info = {
            "model_name": "prophet",
            "model": model,
            "regressors": regressors
        }

        return preds.tolist(), metrics, model_info

    except Exception as e:
        return None, {"mape": 999, "mae": 999, "rmse": 999}, {"error": str(e)}


def forecast_prophet(model_info: dict, df: pd.DataFrame, horizon: int, future_df: Optional[pd.DataFrame] = None):
    model = model_info["model"]
    regressors = model_info.get("regressors", [])

    if future_df is not None:
        future = future_df.rename(columns={"fecha": "ds"}).copy()
    else:
        last_date = df["fecha"].max()

        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1),
            periods=horizon,
            freq="MS"
        )

        future = pd.DataFrame({"ds": future_dates})

    for r in regressors:
        if r in future.columns:
            future[r] = pd.to_numeric(future[r], errors="coerce").fillna(0)
        elif r in df.columns:
            future[r] = float(pd.to_numeric(df[r], errors="coerce").fillna(0).iloc[-1])
        else:
            future[r] = 0

    forecast = model.predict(future[["ds"] + regressors])

    return [float(x) for x in forecast["yhat"].values]


# =====================================================
# MODELO 4: XGBOOST
# =====================================================

def backtest_xgboost(train: pd.DataFrame, test: pd.DataFrame):
    try:
        combined = pd.concat([train, test], ignore_index=True)
        combined_feat, feature_cols, encoder = prepare_ml_features(combined)

        train_end_date = train["fecha"].max()

        train_feat = combined_feat[combined_feat["fecha"] <= train_end_date]

        if len(train_feat) < 6 or len(test) == 0:
            return None, {"mape": 999, "mae": 999, "rmse": 999}, {"error": "Datos insuficientes para XGBoost"}

        X_train = train_feat[feature_cols]
        y_train = train_feat["ventas"]

        model = XGBRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            objective="reg:squarederror",
            n_jobs=1,
            random_state=42
        )

        model.fit(X_train, y_train)

        model_info = {
            "model_name": "xgboost",
            "model": model,
            "feature_cols": feature_cols,
            "encoder": encoder
        }

        # La previsión del test se genera de forma recursiva, con el mismo
        # procedimiento que se emplea en producción y utilizando únicamente
        # las variables exógenas del periodo de test. Así la comparación con
        # SARIMAX y Prophet, que también prevén a varios pasos, es homogénea.
        preds = forecast_xgboost(model_info, train, len(test), test)

        metrics = calculate_metrics(test["ventas"].values, preds)

        return preds, metrics, model_info

    except Exception as e:
        return None, {"mape": 999, "mae": 999, "rmse": 999}, {"error": str(e)}


def forecast_xgboost(model_info: dict, df: pd.DataFrame, horizon: int, future_df: Optional[pd.DataFrame] = None):
    model = model_info["model"]
    feature_cols = model_info["feature_cols"]
    encoder = model_info["encoder"]

    history = df.copy().sort_values("fecha")
    preds = []

    for i in range(horizon):
        last_date = history["fecha"].max()
        next_date = last_date + pd.DateOffset(months=1)

        if future_df is not None and i < len(future_df):
            new_row = history.iloc[-1:].copy()
            for col in ["pedidos_comprometidos", "precio", "promo", "evento", "intensidad_evento"]:
                if col in future_df.columns:
                    new_row[col] = future_df.iloc[i][col]
        else:
            new_row = history.iloc[-1:].copy()

        new_row["fecha"] = next_date
        new_row["ventas"] = np.nan

        history_temp = pd.concat([history, new_row], ignore_index=True)

        # dropna_target=False conserva la fila del periodo a predecir
        # (su "ventas" es NaN); en caso contrario se eliminaría y se
        # predeciría con las variables del mes anterior.
        feat_df, _, _ = prepare_ml_features(
            history_temp,
            encoder=encoder,
            dropna_target=False
        )

        next_features = feat_df.iloc[-1:][feature_cols]

        pred = float(model.predict(next_features)[0])
        preds.append(pred)

        new_row["ventas"] = pred
        history = pd.concat([history, new_row], ignore_index=True)

    return preds


# =====================================================
# BACKTESTING Y SELECCIÓN DE MODELO
# =====================================================

def split_train_test(df: pd.DataFrame):
    df = df.sort_values("fecha").copy()

    if len(df) < 8:
        raise HTTPException(
            status_code=400,
            detail="Se necesitan al menos 8 meses de datos para comparar modelos."
        )

    test_size = max(3, int(len(df) * 0.25))

    if len(df) - test_size < 5:
        test_size = max(1, len(df) - 5)

    train = df.iloc[:-test_size].copy()
    test = df.iloc[-test_size:].copy()

    return train, test


def run_model_selection(sku: str):
    df = get_sku_data(sku)

    train, test = split_train_test(df)

    results = []

    naive_preds, naive_metrics, naive_model = backtest_seasonal_naive(train, test)
    results.append({
        "model_name": "seasonal_naive",
        "metrics": naive_metrics,
        "model_info": naive_model
    })

    sarimax_preds, sarimax_metrics, sarimax_model = backtest_sarimax(train, test)
    results.append({
        "model_name": "sarimax",
        "metrics": sarimax_metrics,
        "model_info": sarimax_model
    })

    prophet_preds, prophet_metrics, prophet_model = backtest_prophet(train, test)
    results.append({
        "model_name": "prophet",
        "metrics": prophet_metrics,
        "model_info": prophet_model
    })

    xgb_preds, xgb_metrics, xgb_model = backtest_xgboost(train, test)
    results.append({
        "model_name": "xgboost",
        "metrics": xgb_metrics,
        "model_info": xgb_model
    })

    valid_results = [
        r for r in results
        if r["metrics"]["mape"] < 999
    ]

    if not valid_results:
        raise HTTPException(status_code=500, detail="Ningún modelo pudo entrenarse correctamente.")

    best = min(valid_results, key=lambda r: r["metrics"]["mape"])

    best_model_name = best["model_name"]

    # Entrenamiento final con todos los datos
    final_model_info = train_final_model(best_model_name, df)

    model_path = os.path.join(MODEL_DIR, f"{sku}_{best_model_name}.pkl")

    for filename in os.listdir(MODEL_DIR):
        if filename.startswith(f"{sku}_") and filename.endswith(".pkl"):
            old_path = os.path.join(MODEL_DIR, filename)
            if old_path != model_path:
                try:
                    os.remove(old_path)
                except OSError:
                    pass

    joblib.dump(final_model_info, model_path)

    metrics_json = {
        r["model_name"]: r["metrics"]
        for r in results
    }

    execute_sql("""
        INSERT INTO modelos
        (sku, modelo, ruta_modelo, mape, mae, rmse, metricas, ultima_actualizacion)
        VALUES
        (:sku, :modelo, :ruta_modelo, :mape, :mae, :rmse, CAST(:metricas AS JSONB), NOW())
        ON CONFLICT (sku)
        DO UPDATE SET
            modelo = EXCLUDED.modelo,
            ruta_modelo = EXCLUDED.ruta_modelo,
            mape = EXCLUDED.mape,
            mae = EXCLUDED.mae,
            rmse = EXCLUDED.rmse,
            metricas = EXCLUDED.metricas,
            ultima_actualizacion = NOW();
    """, {
        "sku": sku,
        "modelo": best_model_name,
        "ruta_modelo": model_path,
        "mape": best["metrics"]["mape"],
        "mae": best["metrics"]["mae"],
        "rmse": best["metrics"]["rmse"],
        "metricas": json.dumps(metrics_json)
    })

    return {
        "sku": sku,
        "best_model": best_model_name,
        "best_metrics": best["metrics"],
        "all_metrics": metrics_json,
        "model_path": model_path
    }


def train_final_model(model_name: str, df: pd.DataFrame):
    df = df.sort_values("fecha").copy()

    if model_name == "seasonal_naive":
        return {
            "model_name": "seasonal_naive",
            "last_values": df["ventas"].tail(12).tolist()
        }

    if model_name == "sarimax":
        exog_cols = get_exog_columns(df)

        y = df["ventas"].astype(float)

        if exog_cols:
            exog = df[exog_cols].astype(float).fillna(0)
        else:
            exog = None

        seasonal_period = 12 if len(df) >= 18 else 1

        model = SARIMAX(
            y,
            exog=exog,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, seasonal_period) if seasonal_period > 1 else (0, 0, 0, 0),
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        fitted = model.fit(disp=False)

        return {
            "model_name": "sarimax",
            "model": fitted,
            "exog_cols": exog_cols
        }

    if model_name == "prophet":
        if not PROPHET_AVAILABLE:
            raise HTTPException(status_code=500, detail="Prophet no está disponible.")

        df_p = df.rename(columns={"fecha": "ds", "ventas": "y"}).copy()

        regressors = [
            "pedidos_comprometidos",
            "precio",
            "promo",
            "intensidad_evento"
        ]

        regressors = [r for r in regressors if r in df_p.columns]

        for r in regressors:
            df_p[r] = pd.to_numeric(df_p[r], errors="coerce").fillna(0)

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False
        )

        for r in regressors:
            model.add_regressor(r)

        model.fit(df_p[["ds", "y"] + regressors])

        return {
            "model_name": "prophet",
            "model": model,
            "regressors": regressors
        }

    if model_name == "xgboost":
        df_feat, feature_cols, encoder = prepare_ml_features(df)

        if len(df_feat) < 6:
            raise HTTPException(status_code=400, detail="No hay datos suficientes para XGBoost.")

        X = df_feat[feature_cols]
        y = df_feat["ventas"]

        model = XGBRegressor(
            n_estimators=120,
            max_depth=3,
            learning_rate=0.05,
            objective="reg:squarederror",
            n_jobs=1,
            random_state=42
        )

        model.fit(X, y)

        return {
            "model_name": "xgboost",
            "model": model,
            "feature_cols": feature_cols,
            "encoder": encoder
        }

    raise HTTPException(status_code=500, detail=f"Modelo no soportado: {model_name}")

def build_future_features_df(
    df: pd.DataFrame,
    horizon: int,
    future_features: Optional[List[FutureFeature]] = None
) -> pd.DataFrame:
    last_date = df["fecha"].max()

    future_dates = pd.date_range(
        start=last_date + pd.DateOffset(months=1),
        periods=horizon,
        freq="MS"
    )

    defaults = {
        "pedidos_comprometidos": 0,
        "precio": 0,
        "promo": 0,
        "evento": "ninguno",
        "intensidad_evento": 0
    }

    for col in defaults:
        if col in df.columns and col != "evento":
            value = pd.to_numeric(df[col], errors="coerce").fillna(defaults[col]).iloc[-1]
            defaults[col] = float(value)

    if "evento" in df.columns:
        defaults["evento"] = str(df["evento"].fillna("ninguno").iloc[-1])

    future_df = pd.DataFrame({"fecha": future_dates})

    for col, value in defaults.items():
        future_df[col] = value

    if future_features:
        provided = pd.DataFrame([
            f.model_dump() if hasattr(f, "model_dump") else f.dict()
            for f in future_features
        ])
        provided["fecha"] = pd.to_datetime(provided["fecha"])

        future_df = future_df.merge(
            provided,
            on="fecha",
            how="left",
            suffixes=("", "_provided")
        )

        for col in defaults:
            provided_col = f"{col}_provided"
            if provided_col in future_df.columns:
                future_df[col] = future_df[provided_col].combine_first(future_df[col])
                future_df = future_df.drop(columns=[provided_col])

    return future_df


# =====================================================
# FORECAST
# =====================================================

def load_saved_model_for_sku(sku: str):
    model_df = read_sql("""
        SELECT *
        FROM modelos
        WHERE sku = %(sku)s;
    """, {"sku": sku})

    if model_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No hay modelo seleccionado para SKU {sku}. Ejecuta /select_model primero."
        )

    model_path = model_df.iloc[0]["ruta_modelo"]

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"No existe el archivo del modelo: {model_path}. Ejecuta /select_model otra vez."
        )

    return joblib.load(model_path), model_df.iloc[0].to_dict()

def generate_forecast(
    sku: str,
    horizon: int,
    future_features: Optional[List[FutureFeature]] = None
):
    df = get_sku_data(sku)

    model_info, model_record = load_saved_model_for_sku(sku)
    future_df = build_future_features_df(df, horizon, future_features)

    model_name = model_info["model_name"]

    if model_name == "seasonal_naive":
        preds = forecast_seasonal_naive(model_info, horizon)

    elif model_name == "sarimax":
        preds = forecast_sarimax(model_info, df, horizon, future_df)

    elif model_name == "prophet":
        preds = forecast_prophet(model_info, df, horizon, future_df)

    elif model_name == "xgboost":
        preds = forecast_xgboost(model_info, df, horizon, future_df)

    else:
        raise HTTPException(status_code=500, detail=f"Modelo no soportado: {model_name}")

    future_dates = future_df["fecha"]

    forecast_rows = []

    for forecast_date, value in zip(future_dates, preds):
        forecast_rows.append({
            "sku": sku,
            "fecha_predicha": forecast_date.date().isoformat(),
            "forecast": max(0, float(value)),
            "modelo_usado": model_name
        })

    return forecast_rows, model_name


def save_forecasts(sku: str, forecast_rows: List[dict]):
    for row in forecast_rows:
        execute_sql("""
            INSERT INTO forecasts
            (sku, fecha_forecast, fecha_predicha, forecast, modelo_usado)
            VALUES
            (:sku, CURRENT_DATE, :fecha_predicha, :forecast, :modelo_usado)
            ON CONFLICT (sku, fecha_forecast, fecha_predicha)
            DO UPDATE SET
                forecast = EXCLUDED.forecast,
                modelo_usado = EXCLUDED.modelo_usado,
                created_at = NOW();
        """, {
            "sku": sku,
            "fecha_predicha": row["fecha_predicha"],
            "forecast": row["forecast"],
            "modelo_usado": row["modelo_usado"]
        })


# =====================================================
# ENDPOINTS
# =====================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "prophet_available": PROPHET_AVAILABLE
    }


@app.post("/select_model")
def select_model(request: SKURequest):
    """
    Hace backtesting con los 4 modelos y guarda el mejor modelo para el SKU.
    Este endpoint es pesado. No se ejecuta cada mes salvo que tú quieras.
    """
    return run_model_selection(request.sku)


@app.post("/retrain")
def retrain(request: SKURequest):
    """
    Reentrena y vuelve a seleccionar el mejor modelo usando los últimos 24 meses.
    En esta versión hace lo mismo que /select_model.
    """
    return run_model_selection(request.sku)


@app.post("/forecast")
def forecast(request: ForecastRequest):
    """
    Genera forecast usando el modelo ya seleccionado.
    No hace backtesting.
    """
    forecast_rows, model_name = generate_forecast(
        request.sku,
        request.horizon,
        request.future_features
    )

    if request.save_forecast:
        save_forecasts(request.sku, forecast_rows)

    return {
        "sku": request.sku,
        "modelo_usado": model_name,
        "horizon": request.horizon,
        "forecast": forecast_rows
    }


@app.post("/evaluate")
def evaluate(request: EvaluateRequest):
    """
    Compara con las ventas reales todas las previsiones que ya tienen dato
    observado y que todavía no han sido evaluadas, y registra el error.

    Para cada mes previsto se evalúa la previsión más reciente disponible.
    Si no hay nada pendiente devuelve una lista vacía, no un error, para que
    el flujo pueda continuar en la primera ejecución o cuando aún no existan
    ventas del periodo previsto.

    error_abs y error_pct se almacenan en valor absoluto, de modo que la media
    de la tabla es directamente el MAE y el MAPE. El sesgo puede obtenerse a
    partir de las columnas real y forecast, que se guardan sin transformar.
    """
    sku = request.sku

    query = """
        SELECT
            f.sku,
            f.fecha_predicha,
            f.forecast,
            v.ventas AS real
        FROM forecasts f
        JOIN ventas v
            ON f.sku = v.sku
            AND f.fecha_predicha = v.fecha
        WHERE f.sku = %(sku)s
        AND f.fecha_forecast = (
            SELECT MAX(f2.fecha_forecast)
            FROM forecasts f2
            WHERE f2.sku = f.sku
            AND f2.fecha_predicha = f.fecha_predicha
        )
        AND NOT EXISTS (
            SELECT 1
            FROM errores e
            WHERE e.sku = f.sku
            AND e.fecha = f.fecha_predicha
        )
        ORDER BY f.fecha_predicha ASC;
    """

    df = read_sql(query, {"sku": sku})

    evaluaciones = []

    for _, row in df.iterrows():
        forecast_value = float(row["forecast"])
        real_value = float(row["real"])

        error_abs = abs(real_value - forecast_value)
        error_pct = error_abs / real_value if real_value != 0 else 0

        execute_sql("""
            INSERT INTO errores
            (sku, fecha, forecast, real, error_abs, error_pct)
            VALUES
            (:sku, :fecha, :forecast, :real, :error_abs, :error_pct);
        """, {
            "sku": sku,
            "fecha": row["fecha_predicha"],
            "forecast": forecast_value,
            "real": real_value,
            "error_abs": error_abs,
            "error_pct": error_pct
        })

        evaluaciones.append({
            "fecha": str(row["fecha_predicha"]),
            "forecast": forecast_value,
            "real": real_value,
            "error_abs": error_abs,
            "error_pct": error_pct
        })

    return {
        "sku": sku,
        "evaluados": len(evaluaciones),
        "detalle": evaluaciones
    }


@app.get("/models/{sku}")
def get_model_info(sku: str):
    df = read_sql("""
        SELECT *
        FROM modelos
        WHERE sku = %(sku)s;
    """, {"sku": sku})

    if df.empty:
        raise HTTPException(status_code=404, detail="No hay modelo para ese SKU.")

    return df.to_dict(orient="records")[0]


@app.get("/errors/{sku}")
def get_errors(sku: str):
    df = read_sql("""
        SELECT *
        FROM errores
        WHERE sku = %(sku)s
        ORDER BY fecha DESC
        LIMIT 18;
    """, {"sku": sku})

    return {
        "sku": sku,
        "errors": df.to_dict(orient="records")
    }

```
