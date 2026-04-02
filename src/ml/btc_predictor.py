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

    def _create_features(self, prices: np.ndarray, volumes: np.ndarray = None) -> pd.DataFrame:

        df = pd.DataFrame({'price': prices})

        if volumes is not None and len(volumes) == len(prices):
            try:
                volumes_num = pd.to_numeric(volumes, errors='coerce')
                if not np.isnan(volumes).all():
                    df['volume'] = volumes
            except Exception as e:
                self.logger.warning(f"No se pudo convertir volumen a numérico: {e}")  
        
        # Returns
        #returns = df['price'].pct_change()
        #df['return_1d'] = returns
        #df['return_7d'] = returns.rolling(7).sum()
        #df['momentum'] = df['price'] / df['price'].shift(7) - 1
#
        ##Volatility
        #df['volatility_7d'] = returns.rolling(7).std()
#
        ## RSI (14 days)
        #delta = df['price'].diff()
        #gain = delta.clip(lower=0).rolling(14).mean()
        #loss = (-delta.clip(upper=0)).rolling(14).mean()
        #rs = gain / loss
        #df['rsi'] = 100 - (100 / (1 + rs))
#
        ## MACD (12,26,9)
        #ema12 = df['price'].ewm(span=12, adjust=False).mean()
        #ema26 = df['price'].ewm(span=26, adjust=False).mean()
        #df['macd'] = ema12 - ema26
        #df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        #df['macd_hist'] = df['macd'] - df['macd_signal']

        # Lags
        for lag in range(1, self.n_lags + 1):
            df[f'lag_{lag}'] = df['price'].shift(lag)

        # Rollings means / stds
        for w in self.windows:
            df[f'rolling_mean_{w}'] = df['price'].shift(1).rolling(w).mean() # .shift(1) to avoid data leakage
            df[f'rolling_std_{w}'] = df['price'].shift(1).rolling(w).std()

        self.logger.info(f"Before dropna: shape={df.shape}")
        nan_counts = df.isna().sum()
        self.logger.info("NaN counts per column:\n" + "\n".join([f"  {col}: {count}" for col, count in nan_counts[nan_counts > 0].items()]))

        # Delete NaN rows
        df.dropna(inplace=True)

        self.logger.info(f"After dropna: shape={df.shape}")
        if df.empty:
            self.logger.warning("DataFrame is empty after dropna!")

        return df



    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        prices = df['price_usd'].values
        volumes = df['volume_usd'].values if 'volume_usd' in df.columns else None
        min_required = self.n_lags + 1

        #DEBUGG
        self.logger.info(f"Input prices shape: {len(prices)}, NaN count: {np.isnan(prices).sum()}")

        if len(prices) < min_required:
            raise ValueError(f"Not enough data: need at least {min_required} prices, got {len(prices)}")

       
        last_prices = prices[-self.n_lags:].copy()
        self.last_prices = last_prices

        feature_df = self._create_features(prices,volumes)
        if len(feature_df) == 0:
            self.logger.error(f"Feature creation failed. Input prices length: {len(prices)}")
            self.logger.error(f"First 10 prices: {prices[:10]}")
            self.logger.error(f"NaN count in prices: {np.isnan(prices).sum()}")
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

