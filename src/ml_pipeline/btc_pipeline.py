import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional, List

class BTCPredictor:

    def __init__(self, model: Optional[LinearRegression] = None):

        self.model = model or LinearRegression()
        self.logger = logging.getLogger(__name__)
        self.is_trained = False 
        self.training_size = 0 
    

    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:

        prices = df["price_usd"].values
        X = np.arange(len(prices)).reshape(-1, 1)
        y = prices

        self.training_size = len(X)
        self.logger.info(f"Prepared data: {self.training_size} samples")
        return X, y
    
    