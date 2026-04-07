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

    def __init__(self, model=None,n_lags: int = 3, windows: List[int] = None, target_type='return'):
        self.model = model or LinearRegression()
        self.n_lags = n_lags
        self.windows = []
        self.target_type = target_type
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = None
        self.last_prices = None
        self.logger = logging.getLogger(__name__)

    def _create_features(self, prices: np.ndarray, volumes: np.ndarray = None) -> pd.DataFrame:
        df = pd.DataFrame({'price': prices})        
        # Logaritmic return
        df['return'] = np.log(df['price'] / df['price'].shift(1))
        # Return's lags
        for lag in range(1, self.n_lags + 1):
            df[f'return_lag_{lag}'] = df['return'].shift(lag)
        # Delete rows with NaN (first n_lags+1 rows)
        df.dropna(inplace=True)
        return df


    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        prices = df['price_usd'].values
        if len(prices) < self.n_lags + 2:
            raise ValueError(f"Need at least {self.n_lags+2} prices, got {len(prices)}")

        # Create  features
        feature_df = self._create_features(prices)
        # Target: 'return'
        y = feature_df['return'].values
        # Features: all the column except 'price' and 'return'
        self.feature_names = [col for col in feature_df.columns if col not in ['price', 'return']]
        X = feature_df[self.feature_names].values
        # Save the last n_lags for prediction
        self.last_prices = prices[-self.n_lags:].copy()
        self.logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
        return X, y, self.last_prices


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
        self.logger.info(f"Features usadas ({len(self.feature_names)}): {self.feature_names}")
        self.logger.info(f"X shape: {X.shape} | y shape: {y.shape}")
        self.logger.info(f"y stats — mean: {y.mean():.1f}, std: {y.std():.1f}, min: {y.min():.1f}, max: {y.max():.1f}")


    def predict_future(self, n_days: int, last_prices=None) -> np.ndarray:
        prices_to_use = last_prices if last_prices is not None else self.last_prices
        if prices_to_use is None:
            raise ValueError("No last_prices available")
        if len(prices_to_use) < self.n_lags + 1:
            raise ValueError(f"Need at least {self.n_lags+1} prices, got {len(prices_to_use)}")
        predictions = []
        current_prices = list(prices_to_use)

        for step in range(n_days):

            returns_series = np.log(np.array(current_prices[1:])) / np.array(current_prices[:-1])

            if len(returns_series) < self.n_lags:
                # si no hay suficientes retornos, rellenar con ceros (sólo ocurre al inicio)
                features = np.zeros(self.n_lags)
                features[-len(returns_series):] = returns_series
            else:
                features = returns_series[-self.n_lags:]
            features = features.reshape(1, -1)

            features_scaled = self.scaler.transform(features)
            pred_return = self.model.predict(features_scaled[0])

            next_price = current_prices[-1] * np.exp(pred_return)
            predictions.append(next_price)
            current_prices.append(next_price)

        
        return np.array(predictions)


