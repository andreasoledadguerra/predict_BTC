import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from typing import Optional, List, Union


class BTCPredictor:
    """
    Bitcoin price predictor with feature engineering:
    - Lags of the price (AR terms)
    - Rolling means and standard deviations
    """

    def __init__(self, model: Optional[Union[LinearRegression, Ridge]] = None,
                 n_lags: int = 7, windows: List[int] = None):
        self.model = model or LinearRegression()
        self.n_lags = n_lags
        self.windows = windows or [7, 14]
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        self.last_prices = None
        self.logger = logging.getLogger(__name__)

    def _create_features(self, prices: np.ndarray) -> pd.DataFrame:

        df = pd.DataFrame({'price': prices})

        for lag in range(1, self.n_lags + 1):
            df[f'lag_{lag}'] = df['price'].shift(lag)

        for w in self.windows:
            df[f'rolling_mean_{w}'] = df['price'].shift(1).rolling(window=w, min_periods=1).mean() # .shift(1) for avoid data leakage
            df[f'rolling_std_{w}'] = df['price'].shift(1).rolling(window=w, min_periods=2).std()

        df.dropna(inplace=True)

        return df



    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        prices = df['price_usd'].values
        min_required = self.n_lags + 1
        if len(prices) < min_required:
            raise ValueError(f"Not enough data: need at least {min_required} prices, got {len(prices)}")

       
        last_prices = prices[-self.n_lags:].copy()
        self.last_prices = last_prices

        feature_df = self._create_features(prices)
        if len(feature_df) == 0:
           raise ValueError(f"Feature creation failed: need at least {min_required} prices, got {len(prices)}")

        self.feature_names = [col for col in feature_df.columns if col != 'price']
        X = feature_df.drop('price', axis=1).values
        y = feature_df['price'].values
       
        self.logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
        return X, y, last_prices


    def train(self, X, y):

        if not self.is_trained:
            # Fit scaler
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)

        # Train model
        self.model.fit(X_scaled, y)

        # ========== SOLO CREAR feature_names SI NO EXISTEN ==========
        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]

        self.is_trained = True
        r2 = self.model.score(X_scaled, y)
        self.logger.info(f"Model trained. R² = {r2:.4f}")


    def predict_future(self, n_days: int, last_prices=None) -> np.ndarray:
        if last_prices is not None:
            prices_to_use = last_prices
        else:
            if self.last_prices is None:
                raise ValueError("No last_prices available")
            prices_to_use = self.last_prices

        min_required = self.n_lags + max(self.windows)
        if len(prices_to_use) < min_required:
            raise ValueError(f"Not enough historical prices to start prediction. Need at least {min_required}, got {len(prices_to_use)}")
        
        predictions = []
        current_prices = list(prices_to_use)

        for step in range(n_days):
            # Crear Series de precios para usar _create_features
            #prices_series = pd.Series(current_prices)

            
            feature_df = self._create_features(np.array(current_prices))

            # Take the last row (that correspond to the last price)
            if len(feature_df) == 0:
                raise ValueError("Cannot create features for prediction at step {step}")

            #Retrieve features from the last row (without the column 'price')
            feature_row = feature_df.drop('price', axis=1).iloc[-1:].values

            # Verify dimensions
            if feature_row.shape[1] != len(self.feature_names):
                #self.logger.error(f"❌ Feature mismatch: got {feature_row.shape[1]}, expected {len(self.feature_names)}")
                #self.logger.error(f"   Created columns: {feature_df.drop('price', axis=1).columns.tolist()}")
                #self.logger.error(f"   Expected: {self.feature_names}")
                raise ValueError(f"Feature count mismatch at step {step}: "
                                 f"got {feature_row.shape[1]}, expected {len(self.feature_names)}"
                                 )

            # Scale and predict
            feature_vector_scaled = self.scaler.transform(feature_row)
            next_price = self.model.predict(feature_vector_scaled)[0]

            predictions.append(next_price)
            current_prices.append(next_price)

        return np.array(predictions)

