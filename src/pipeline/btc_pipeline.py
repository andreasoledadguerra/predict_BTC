
import pandas as pd
import logging
from sklearn.linear_model import Ridge
from src.api.coingecko_client import CoinGeckoClient
from src.database.postgres_manager import DatabaseManager
from src.ml.btc_predictor import BTCPredictor

logger = logging.getLogger(__name__)

class BTCDataPipeline:
    """
    Pipeline for fetching, storing, and retrieving BTC data.
    """

    def __init__(self, api_client: CoinGeckoClient, db_manager: DatabaseManager, predictor: BTCPredictor):
        self.api_client = api_client
        self.db_manager = db_manager
        self.predictor = predictor

    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch BTC prices from CoinGecko API."""
        logger.info(f"Fetching data from {start_date} to {end_date} from CoinGecko...")
        df = self.api_client.fetch_bitcoin_prices(start_date, end_date)
        return df

    def save_data_in_db(self, df: pd.DataFrame) -> int:
        """Save BTC prices to database."""
        logger.info("Saving data to database...")
        records = self.db_manager.save_btc_prices(df)
        return records

    def get_data_for_training(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Retrieve historical BTC prices from database for training."""
        logger.info(f"Retrieving data from {start_date} to {end_date} from database...")
        df = self.db_manager.get_btc_prices(start_date, end_date)
        #DEBUG
        #self.logger.info(f"Columns in DataFrame: {df.columns.tolist()}")
        return df
    
    def predict_training_data(
        self, 
        df, 
        n_days_future,
        alpha=1.0
    ):
        """Train Linear and Ridge models, and generate predictions"""
        # ---- PREPARE DATA (for both models) ----

        temp_predictor = BTCPredictor()
        X, y, last_prices = temp_predictor.prepare_training_data(df)

        #context window
        extended_prices = df['price_usd'].values[-20:]
        
        # ---- MODELO 1: LINEAR REGRESSION ----
        linear_predictor = BTCPredictor()

        
        linear_predictor.feature_names = temp_predictor.feature_names
        linear_predictions = linear_predictor.predict_future(n_days_future, last_prices=extended_prices, )
        X_scaled =linear_predictor.scaler.transform(X)
        linear_r2 = linear_predictor.model.score(X_scaled, y)
        
        # ---- MODELO 2: RIDGE REGRESSION ----
        ridge_predictor = BTCPredictor(model=Ridge(alpha=alpha))

        ridge_predictor.feature_names = temp_predictor.feature_names
        ridge_predictor.train(X, y)
        ridge_predictions = ridge_predictor.predict_future(n_days_future,last_prices=extended_prices)
        X_scaled =ridge_predictor.scaler.transform(X)
        ridge_r2 = ridge_predictor.model.score(X_scaled, y)
        
        # ---- LOGGING ----
        logger.info(f"✅ Linear Regression trained: R² = {linear_r2:.4f}")
        logger.info(f"✅ Ridge Regression trained: R² = {ridge_r2:.4f}, α = {alpha}")
        
        # ---- RETURN DICT ----
        return {
            'linear': linear_predictions, # predictions Linear
            'ridge': ridge_predictions,  # predictions_Ridge
            'linear_r2': linear_r2,  #  R² score Linear
            'ridge_r2': ridge_r2,   # R² score Ridge
            'linear_model': linear_predictor, # Trained Linear model
            'ridge_model': ridge_predictor, # Trained Ridge model
            'last_prices': last_prices
        }