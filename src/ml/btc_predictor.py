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

        for w in self.windows:
            df[f'return_rolling_mean_{w}'] = df['return'].shift(1).rolling(w).mean()
            df[f'return_rolling_std_{w}']  = df['return'].shift(1).rolling(w).std()

        # Volatilidad realizada
        df['volatility_7d'] = df['return'].shift(1).rolling(7).std()
        df['volatility_14d'] = df['return'].shift(1).rolling(14).std()

        # Momentum
        df['momentum_7d']  = df['price'].shift(1) / df['price'].shift(8)  - 1
        df['momentum_14d'] = df['price'].shift(1) / df['price'].shift(15) - 1

        # RSI (14 días) — sobre retornos
        delta = df['price'].diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'].shift(1)  # evitar leakage

        # MACD
        ema12 = df['price'].ewm(span=12, adjust=False).mean()
        ema26 = df['price'].ewm(span=26, adjust=False).mean()
        df['macd']        = (ema12 - ema26).shift(1)
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist']   = df['macd'] - df['macd_signal']

        # Volumen (si está disponible y no es todo NaN)
        if volumes is not None and len(volumes) == len(prices):
            vol_series = pd.to_numeric(
                pd.Series(volumes, index=df.index), errors='coerce'
            )
            if not vol_series.isna().all():
                df['volume'] = vol_series
                df['volume_change'] = vol_series.pct_change().shift(1)

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
    
        # Función para aplanar y convertir a float cualquier estructura anidada
        def flatten_to_floats(obj):
            result = []
            if isinstance(obj, (list, tuple, np.ndarray)):
                for item in obj:
                    result.extend(flatten_to_floats(item))
            else:
                try:
                    result.append(float(obj))
                except (ValueError, TypeError):
                    raise ValueError(f"Non-numeric item found: {obj}")
            return result
    
        current_prices = flatten_to_floats(prices_to_use)

        predictions = []
        for step in range(n_days):
            prices_arr = np.array(current_prices, dtype=float)

            feature_df = self._create_features(prices_arr)

            if len(feature_df) == 0:
                raise ValueError(f"No features generadas en step {step}. "
                           f"Precios disponibles: {len(prices_arr)}")

            # Verificar que todas las feature_names están presentes
            missing = [f for f in self.feature_names if f not in feature_df.columns]
            if missing:
                raise ValueError(f"Features faltantes en step {step}: {missing}")

             # Tomar la última fila — corresponde al último precio disponible
            feature_row = feature_df[self.feature_names].iloc[-1].values.reshape(1, -1)

            self.logger.info(f"Step {step}: precios={len(prices_arr)}, "
                        f"features={feature_row.shape}, "
                        f"columnas={feature_df.columns.tolist()}")

            features_scaled = self.scaler.transform(feature_row)
            pred_return = self.model.predict(features_scaled)[0]
            next_price = float(current_prices[-1] * np.exp(pred_return))
         
            predictions.append(next_price)
            current_prices.append(next_price)
    
        return np.array(predictions)


