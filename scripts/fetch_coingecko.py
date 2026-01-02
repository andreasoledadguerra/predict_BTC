import os
import requests
import pandas as pd

# Standard library imports for date and time handling
from datetime import datetime, timedelta, timezone

from typing import Tuple

# Third-party library to load environment variables form a .env file
from dotenv import load_dotenv

from sqlalchemy import create_engine

engine = create_engine(
    "postgresql+psycopg2://andy:secret123@localhost:5432/crypto"
)

# Load variables defined in the .env file into the environment
# making them accessible via os.getenv()
load_dotenv()


# Retrieve the CoinGecko API key from environment variables
API_KEY = os.getenv("COINGECKO_API_KEY")

# Retireve the Postgres environment variables
POSTGRES_USER=     os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD= os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB=      os.getenv("POSTGRES_DB")
POSTGRES_PORT=     os.getenv("POSTGRES_PORT")

# Define the time range for the API request

def str_to_timestamp(date_str:str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def fetch_date(start_date: str, end_date: str) -> Tuple [int,int]:
    return (
        str_to_timestamp(start_date),
        str_to_timestamp(end_date),
    )


#start_ts, end_ts = fetch_date(start_date, end_date)


# CoinGecko endpoint for retrieving price data within a time range
url =  "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"


# Query parameterss sent in the request URL
params = {
    "vs_currency": "usd",
    "from": start_ts,
    "to": end_ts
}

# HTTP headers, including the demo API key
headers = {
    "accept": "application/json",
    "x-cg-demo-api-key": API_KEY
}

# Perform the API request
response = requests.get(url, params=params, headers=headers)

# Parse the JSON response inti a Python dictionary
data = response.json()

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
        conn.execute("SELECT 1")
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
