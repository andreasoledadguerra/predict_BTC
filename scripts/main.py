import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv


from api.coingecko_client import CoinGeckoClient
from database.postgres_manager import DatabaseManager
from ml.btc_predictor import BTCPredictor
from pipeline.btc_pipeline import BTCDataPipeline

# Load variables defined in the .env file into the environment (delete)
load_dotenv()

# Retrieve the Postgres environment variables (delete)
POSTGRES_USER= os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD= os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB= os.getenv("POSTGRES_DB")
POSTGRES_PORT= os.getenv("POSTGRES_PORT")

# Retrieve the CoinGecko API key from environment variables(delete)
API_KEY = os.getenv("COINGECKO_API_KEY")