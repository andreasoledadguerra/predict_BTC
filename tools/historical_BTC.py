# script para cargar precio histórico del BTC

import pandas as pd
import sqlite3 

from dotenv import load_dotenv
from src.api.coingecko_client import CoinGeckoClient
from src.database.postgres_manager import DatabaseManager
from src.ml.btc_predictor import BTCPredictor
from src.pipeline.btc_pipeline import BTCDataPipeline
from src.config.settings import get_postgres_settings, get_coingecko_settings

