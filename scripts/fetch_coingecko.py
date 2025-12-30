import os
import requests
import pandas as pd

# Standard library imports for date and time handling
from datetime import datetime, timedelta, timezone

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


# Define the time range for the API request
end_ts = int(datetime.now(timezone.utc).timestamp())
start_ts = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp())

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


# Basic output for inspection and debugging
print("Keys:", data.keys())
print("First price entry:", data["prices"][0])
