import os
import requests
import pandas as pd
import numpy as np

from datetime import datetime, timezone
from typing import Tuple
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load variables defined in the .env file into the environment (borrar)
load_dotenv()

# Retireve the Postgres environment variables (borrar)
POSTGRES_USER= os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD= os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB= os.getenv("POSTGRES_DB")
POSTGRES_PORT= os.getenv("POSTGRES_PORT")

# Retrieve the CoinGecko API key from environment variables
API_KEY = os.getenv("COINGECKO_API_KEY")

# Create conection to database
engine = create_engine(
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:{POSTGRES_PORT}/{POSTGRES_DB}"
)

def str_to_timestamp(date_str:str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())

# Convert date string to timestamp Unix
def convert_to_unix(start_date:str, end_date: str) -> tuple[int, int]:
    start_ts = str_to_timestamp(start_date)
    end_ts =str_to_timestamp(end_date)
    return start_ts, end_ts

def build_coingecko_request(start_ts: int, end_ts: int) -> dict:
    return {
        "url":  "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range",
        "params": {      
            "vs_currency": "usd",
            "from": start_ts,
            "to": end_ts
        },                         
        "headers": {
            "x-cg-demo-api-key": API_KEY
        }
    }

def fetch_from_api(url: str, params: dict, headers: dict) -> dict:
    response = requests.get(url, params=params, headers=headers)

    if response.status_code != 200:
        raise Exception(f"API Error: {response.status_code}")
    
    return response.json() # Parse the JSON response into a Python dictionary

def parse_price_data(data:dict) -> pd.DataFrame:

    if "prices" not in data:
        raise ValueError ("La respuesta no contiene 'prices'")
    
    # Create basic DataFrame
    df = pd.DataFrame(data["prices"], columns=["timestamp_ms", "price_usd"])

    # Transformar timestamp a fecha
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.date

    # Agregar columna de activo
    df["asset"] = "BTC"

    # Reordenar columnas
    df = df[["date", "proce_usd", "asset"]]

    return df

# Get data from CoinGecko's API
def fetch_date(start_date: str, end_date: str) -> pd.DataFrame:
    
    # CoinGecko endpoint for retrieving price data within a time rang

    # Perform the API request
  

    df = pd.DataFrame(data["prices"], columns=["timestamp_ms", "price_usd"])
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.date
    df["asset"] = "BTC"
    df = df[["date", "price_usd", "asset"]]
    
    print(f"{len(df)} registros obtenidos")

    return df

# Guardar en Postgres
def save_to_database(df: pd.DataFrame):
    print("\nGuardando en la base de datos...")

    try:
        df.to_sql(
            "btc_prices",
            engine,
            if_exists="append", # no borra datos anteriores
            index=False,
            method='multi' # for better performance
        )
        print(f" {len(df)} registros guardados exitosamente")
    except Exception as e:
        print(f"Error al guardar: {e}")
        raise

# Get data from database
def get_data_from_db(start_date:str, end_date:str) -> pd.DataFrame:
    query = """
    SELECT date, price_usd 
    FROM btc_prices 
    WHERE date BETWEEN %s AND %s 
    ORDER BY date
    """
    df = pd.read_sql(query, engine, params=(start_date, end_date))
    return df

# Set data for regression
def train_and_predict(df: pd.DataFrame, future_days: int):
    prices = df["price_usd"].values
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices

    # Training linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Definir el rango de predicción

    X_future = np.arange(len(prices), len(prices) + future_days).reshape(-1, 1)
    predictions = model.predict(X_future)

    return predictions


# -----------------------------------------------------------

def main():
    print("=" * 70)
    print("BITCOIN PRICE PREDICTOR")
    print("=" * 70)
    
    
    # 1: OBTENER Y GUARDAR DATOS NUEVOS
    print("\nObtener datos nuevos de CoinGecko")
    print("-" * 70)
    
    print("\nIngrese el rango de fechas para DESCARGAR datos nuevos:")
    fetch_start = input("  Fecha inicial (YYYY-MM-DD): ")
    fetch_end = input("  Fecha final (YYYY-MM-DD): ")
    
    # Obtener datos de la API
    df_new = fetch_date(fetch_start, fetch_end)
    print("\nPreview de los datos obtenidos:")
    print(df_new.head())
    
    # Guardar en la base de datos
    save_to_database(df_new)
    
    
    # 2: LEER DATOS DE LA DB Y PREDECIR
    print("\n" + "=" * 70)
    print("Predecir precios usando datos de la base de datos")
    print("-" * 70)
    
    print("\nIngrese el rango de fechas para ENTRENAR el modelo:")
    print("(Puede ser el mismo rango u otro diferente)")
    train_start = input("  Fecha inicial (YYYY-MM-DD): ")
    train_end = input("  Fecha final (YYYY-MM-DD): ")
    
    # LEER desde la base de datos
    df_train = get_data_from_db(train_start, train_end)
    
    if len(df_train) == 0:
        print(" No hay datos en la base de datos para ese rango")
        return
    
    print("\nPreview de los datos de entrenamiento:")
    print(df_train.head())

    print(f"Registros usados para entrenar: {len(df_train)}")
    print(df_train.tail())

    try:
        days_input = input("\nElija cuántos días quiere predecir (Si apreta Enter serán 10): ")
        future_days = int(days_input) if days_input.strip() else 10
    except ValueError:
        print(" Entrada inválida, usando 10 días")
        future_days = 10

    predictions = train_and_predict(df_train, future_days)

    print("\n Predicciones:")
    for i, pred in enumerate(predictions, 1):
        print(f"   Día {i}: ${pred:,.2f}")


    print("\n" + "=" * 70)
    print(" Proceso completado exitosamente")
    print("=" * 70)



# EJECUTAR
if __name__ == "__main__":
        main()
