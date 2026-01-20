
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import pandas as pd


from api.coingecko_client import CoinGeckoClient
from database.postgres_manager import DatabaseManager
from ml.btc_predictor import BTCPredictor


