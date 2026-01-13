import os
import requests
import pandas as pd
import numpy as np

from datetime import datetime, timezone
from typing import Tuple
from sklearn.linear_model import LinearRegression
from dotenv import load_dotenv
from sqlalchemy import create_engine

# Load variables defined in the .env file into the environment (delete)
load_dotenv()

# Retireve the Postgres environment variables (delete)
POSTGRES_USER= os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD= os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB= os.getenv("POSTGRES_DB")
POSTGRES_PORT= os.getenv("POSTGRES_PORT")

# Retrieve the CoinGecko API key from environment variables(delete)
API_KEY = os.getenv("COINGECKO_API_KEY")

# Create conection to database(delete)
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
    
    df = pd.DataFrame(data["prices"], columns=["timestamp_ms", "price_usd"])

    # Transformar timestamp a fecha (borrar)
    df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.date

    # Agregar columna de activo
    df["asset"] = "BTC"

    # Reordenar columnas
    df = df[["date", "price_usd", "asset"]]

    return df

# Get data from CoinGecko's API (delete)
def fetch_bitcoin_prices(start_date: str, end_date: str) -> pd.DataFrame:
    start_ts, end_ts = convert_to_unix(start_date, end_date)
    request_config = build_coingecko_request(start_ts, end_ts)

    print(f"Obteniendo datos desde {start_date} hasta {end_date}...")
    raw_data = fetch_from_api(
        request_config["url"],
        request_config["params"],
        request_config["headers"]
    )

    df = parse_price_data(raw_data)

    print(f" {len(df)} registros obtenidos")
    return df
        

def save_to_database(df: pd.DataFrame):
    print("\nGuardando en la base de datos...")

    try:
        df.to_sql(
            "btc_prices",
            engine,
            if_exists="append",
            index=False,
            method='multi' # for better performance
        )
        print(f" {len(df)} registros guardados exitosamente")
    except Exception as e:
        print(f"Error al guardar: {e}")
        raise


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

#def train_and_predict(df: pd.DataFrame, future_days: int):
#    prices = df["price_usd"].values
#    X = np.arange(len(prices)).reshape(-1, 1)
#    y = prices
#
#    # Training linear regression
#    model = LinearRegression()
#    model.fit(X, y)
#
#    # Define date range
#    X_future = np.arange(len(prices), len(prices) + future_days).reshape(-1, 1)
#    predictions = model.predict(X_future)
#
#    return predictions

def prepare_training_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    prices = df["price_usd"].values
    X = np.arange(len(prices)).reshape(-1, 1)
    y = prices

    return X, y


def train_linear_model(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    model = LinearRegression()
    model.fit(X, y)

    return model


def generate_future_data(n_current: int, n_future:int) -> np.ndarray:
    X_future = np.arange(n_current, n_current + n_future).reshape(-1, 1)

    return X_future


def make_predictions(model: LinearRegression, X_future: np.ndarray) -> np.ndarray:
    predictions = model.predict(X_future)

    return predictions

# orchestrator function
#def train_and_predict(df: pd.DataFrame) -> np.ndarray:
#    X, y = prepare_training_data(df)       
#    model = train_linear_model(X, y)        
#    X_future = generate_future_data(n_current= int, n_future= int)    
#    predictions = make_predictions(model, X_future)     
#    return predictions

 #------------------------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("BITCOIN PRICE PREDICTOR")
    print("=" * 70)
    
    
    # 1: GET NEW DATA FROM COINGECKO AND SAVE IN DB
    print("\nFetch new data from CoinGecko")
    print("-" * 70)
    
    print("\nEnter the date range to DOWNLOAD new data:")
    fetch_start = input("  Start date (YYYY-MM-DD): ")
    fetch_end = input("  End date (YYYY-MM-DD): ")
    
    df_new = fetch_bitcoin_prices(fetch_start, fetch_end)
    print("\nPreview of fetched data:")
    print(df_new.head())
    
    save_to_database(df_new)
    
    
    # 2: READ DATA FROM DB AND PREDICT
    print("\n" + "=" * 70)
    print("Predict prices using database data")
    print("-" * 70)
    
    print("\nEnter the date rangeto TRAIN the model:")
    print("(It can be the same range or a different one)")
    train_start = input("  Start date (YYYY-MM-DD): ")
    train_end = input("  End date (YYYY-MM-DD): ")
    
    df_train = get_data_from_db(train_start, train_end)
    
    if len(df_train) == 0:
        print(" NO data found in the database for the selected range")
        return
    
    print("\nPreview of training data:")
    print(df_train.head())

    print(f"Records used for training: {len(df_train)}")
    print(df_train.tail())

    try:
        days_input = input("\nChoose how many days you want to predict(Press Enter for 10): ")
        future_days = int(days_input) if days_input.strip() else 10
    except ValueError:
        print(" Invalid input, using 10 days")
        future_days = 10



    #def prepare_training_data(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    #prices = df["price_usd"].values
    #X = np.arange(len(prices)).reshape(-1, 1)
    #y = prices

    #return X, y


    #def train_linear_model(X: np.ndarray, y: np.ndarray) -> LinearRegression:
    #    model = LinearRegression()
    #    model.fit(X, y)
    #
    #    return model


    X, y = prepare_training_data(df_train)


    print(f"Entrenando modelo con {len(X)} datos...")
    model = train_linear_model(X, y )

    X_future = generate_future_data(len(X), future_days)

    predictions = make_predictions(model, X_future)


    print("\n Predictions:")
    for i, pred in enumerate(predictions, 1):
        print(f"   Day {i}: ${pred:,.2f}")


    print("\n" + "=" * 70)
    print(" Process completed successfully")
    print("=" * 70)



# RUN
if __name__ == "__main__":
        main()
