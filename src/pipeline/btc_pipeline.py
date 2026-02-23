
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
        return df
    
    def predict_training_data(
        self, 
        df, 
        start_date, 
        end_date, 
        n_days_future,
        alpha=1.0
    ):
        """
        Entrenar modelos Linear y Ridge, generar predicciones.
        
        Args:
            df: DataFrame con datos históricos
            start_date: Fecha inicio (para logging)
            end_date: Fecha fin (para logging)
            n_days_future: Días a predecir
            alpha: Parámetro de regularización para Ridge
        
        Returns:
            dict: {
                'linear': np.array,       # Predicciones Linear
                'ridge': np.array,        # Predicciones Ridge
                'linear_r2': float,       # R² score Linear
                'ridge_r2': float,        # R² score Ridge
                'linear_model': model,    # Modelo entrenado Linear
                'ridge_model': model      # Modelo entrenado Ridge
            }
        """
        # ---- PREPARE DATA (for both models) ----

        temp_predictor = BTCPredictor()
        X, y, last_prices = temp_predictor.prepare_training_data(df)

        ###
        extended_prices = df['price_usd'].values[-20:]

        logger.info(f"🔍 extended_prices length: {len(extended_prices)}")
        logger.info(f"🔍 extended_prices (primeros 5): {extended_prices[:5]}")
        logger.info(f"🔍 extended_prices (últimos 5): {extended_prices[-5:]}")
        
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
        X_scaled =linear_predictor.scaler.transform(X)
        ridge_r2 = ridge_predictor.model.score(X_scaled, y)
        
        # ---- LOGGING ----
        logger.info(f"✅ Linear Regression trained: R² = {linear_r2:.4f}")
        logger.info(f"✅ Ridge Regression trained: R² = {ridge_r2:.4f}, α = {alpha}")
        
        # ---- RETORNAR DICCIONARIO ----
        return {
            'linear': linear_predictions,
            'ridge': ridge_predictions,
            'linear_r2': linear_r2,
            'ridge_r2': ridge_r2,
            'linear_model': linear_predictor,
            'ridge_model': ridge_predictor,
            'last_prices': last_prices
        }