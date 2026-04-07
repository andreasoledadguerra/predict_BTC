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

        feature_df = self._create_features(np.array(prices_to_use))

        if len(feature_df) == 0:
            raise ValueError("Cannot create features")
        
        feature_row = feature_df[self.feature_names].iloc[-1].values.reshape(1, -1)
        feature_scaled = self.scaler.transform(feature_row)

        if self.target_type == 'diff':
            pred_diff = self.model.predict(feature_scaled)[0]
            final_price = prices_to_use[-1] + pred_diff

            # Interpolate linearly to obtain n_days points for the plot
            predictions = np.linspace(prices_to_use[-1], final_price, n_days + 1)[1:]

        
        return predictions



        #if last_prices is not None:
        #    prices_to_use = last_prices
        #else:
        #    if self.last_prices is None:
        #        raise ValueError("No last_prices available")
        #    prices_to_use = self.last_prices
#
        ## We need at least n_lags + max(windows) + 1 to have an initial difference
        #min_required = self.n_lags + max(self.windows) + 1
        #if len(prices_to_use) < min_required:
        #    raise ValueError(f"Need at least {min_required} prices, got {len(prices_to_use)}")
        #
        #predictions = []
        #current_prices = list(prices_to_use)
#
        #for step in range(n_days):
        #    # Crear Series de precios para usar _create_features        
        #    feature_df = self._create_features(np.array(current_prices))
#
        #    # Take the last row (that correspond to the last price)
        #    if len(feature_df) == 0:
        #        raise ValueError(f"Cannot create features at step {step}")
#
        #    #Retrieve features from the last row (without the column 'price')
        #    feature_row = feature_df[self.feature_names].iloc[-1:].values
        #    feature_scaled = self.scaler.transform(feature_row)
#
        #    if self.target_type == 'diff':
        #        pred_diff = self.model.predict(feature_scaled)[0]
        #        next_price = current_prices[-1] + pred_diff
        #    elif self.target_type == 'return':
        #        pred_return = self.model.predict(feature_scaled)[0]
        #        next_price = current_prices[-1] * np.exp(pred_return)
        #    else:  # price
        #        next_price = self.model.predict(feature_scaled)[0]
        #
        #    predictions.append(next_price)
        #    current_prices.append(next_price)
#
#
        #return np.array(predictions)

