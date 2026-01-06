import os
import requests
import pandas as pd
import numpy as np

# Standard library imports for date and time handling
from datetime import datetime, timezone

from typing import Tuple

# Third-party library to load environment variables form a .env file
from dotenv import load_dotenv

from sqlalchemy import create_engine, text

# Load variables defined in the .env file into the environment
# making them accessible via os.getenv()
load_dotenv()

# Retireve the Postgres environment variables
POSTGRES_USER= os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD= os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB= os.getenv("POSTGRES_DB")
POSTGRES_PORT= os.getenv("POSTGRES_PORT")

engine = create_engine(
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:{POSTGRES_PORT}/{POSTGRES_DB}"
)

# Retrieve the CoinGecko API key from environment variables
API_KEY = os.getenv("COINGECKO_API_KEY")

# Define the time range for the API request

print(" Ingrese el rango de fechas (YYYY-MM-DD)")
start_date_input = input("Fecha inicial: ")
end_date_input = input("Fecha final: ")

def str_to_timestamp(date_str:str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch_date(start_date: str, end_date: str) -> Tuple [int,int]:
    return (
        str_to_timestamp(start_date),
        str_to_timestamp(end_date),
    )

start_ts, end_ts = fetch_date(start_date_input, end_date_input)


# CoinGecko endpoint for retrieving price data within a time range
url =  "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"


# Query parameters sent in the request URL
params = {
    "vs_currency": "usd",
    "from": start_ts,
    "to": end_ts
}

# HTTP headers, including the demo API key
headers = {
    #"accept": "application/json",
    "x-cg-demo-api-key": API_KEY
}

# Perform the API request
response = requests.get(url, params=params, headers=headers)

# Parse the JSON response inti a Python dictionary
data = response.json()

# ========== DEBUG ==========
print("\n" + "="*60)
print("🔍 DEBUG - Respuesta de la API:")
print(f"Status Code: {response.status_code}")
print(f"Keys en respuesta: {list(data.keys())}")
print(f"Respuesta completa: {data}")
print("="*60 + "\n")

# Verificar si hay error
if "error" in data:
    raise Exception(f"❌ API Error: {data['error']}")

if "prices" not in data:
    raise Exception(f"❌ La respuesta no contiene 'prices'. Respuesta: {data}")
# ===================================================

# Process data
df = pd.DataFrame(data["prices"], columns=["timestamp_ms", "price_usd"])
 # ----------------------------------------------------------------------------

df = pd.DataFrame(data["prices"], columns=["timestamp_ms", "price_usd"])

df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms")
df = df.drop(columns="timestamp_ms")

#df = df.rename(columns={"price_usd": "price"})
df["date"] = df["date"].dt.date
df["asset"] = "BTC"
df = df[["date", "price_usd", "asset"]]

print(f"Datos obtenidos: {len(df)} registros")
print(df.head())

# ------------------------------------------------------------------------------
#Test connection
print("\nProbando conexión a PostgresSQL...")
try: 
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
        print("Conexión exitosa a PostgreSQL")
except Exception as e:
    print(f"Error de conexión: {e}")

# Guardar en Postgres
print("\nGuardando en la base de datos...") 
try:
    df.to_sql(
        "btc_prices",
        engine,
        if_exists="replace", #sólo guarda registros nuevos
        index=False,
        method='multi' # for better performance
    )
    print(f" {len(df)} registros guardados exitosamente")
except Exception as e:
    print(f"Error al guardar: {e}")

# ---------------------------------------------------------------------------------
