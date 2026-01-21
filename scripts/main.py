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