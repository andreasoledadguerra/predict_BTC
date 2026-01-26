
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
import pandas as pd
import inspect



from ..api.coingecko_client import CoinGeckoClient
from ..database.postgres_manager import DatabaseManager
from ..ml.btc_predictor import BTCPredictor


class BTCDataPipeline:

    def __init__(self, 
                 api_client: CoinGeckoClient,
                 db_manager: DatabaseManager,
                 predictor: BTCPredictor,
                 ):
                
                self.api_client = api_client
                self.db_manager = db_manager
                self.predictor = predictor
                self.logger = logging.getLogger(__name__)

        
    def fetch_data(
        self,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        self.logger.info(f"Fetching data: {start_date} to {end_date} to CoinGecko API...")
        df_new = self.api_client.fetch_bitcoin_prices(start_date, end_date)
        return df_new

    def save_data_in_db(
        self,
        df_new: pd.DataFrame
    ) -> Dict[str, Any]:
        self.logger.info(f"Saving data in database...")

        records_saved = self.db_manager.save_btc_prices(df_new)
        return records_saved


    def get_data_for_training(
        self,
        train_start_date: str,
        train_end_date: str,
        
    ) -> pd.DataFrame:
        self.logger.info(f"Retrieving training data from PostgreSQL..")

        df_train = self.db_manager.get_btc_prices(
          train_start_date,
          train_end_date,
        )
        return df_train
    

    def predict_training_data(
        self,
        df_train : pd.DataFrame,
        train_start_date: str,
        train_end_date: str,
        predict_days: int = 10
    ) -> Dict[str, Any]:
        self.logger.info(f"Preparing data for prediction...")
        X, y = self.btc_predictor.prepare_training_data(df_train)

        self.logger.info(f"Training linear regression model...")
        self.btc_predictor.train(X, y)
          
    
        self.logger.info(f" Generating predictions from {train_start_date} to {train_end_date}")
        predictions = self.btc_predictor.predict_future(predict_days)
        predictions = predictions.tolist()
    
        return predictions