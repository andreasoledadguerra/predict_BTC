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
                 n_lags: int = 3, windows: List[int] = None, target_type='return'):
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
        if volumes is not None and len(volumes) == len(prices) and not np.all(np.isnan(volumes)):
            df['volume'] = pd.to_numeric(volumes, errors='coerce')

        # Logaritmic returns
        df['return'] = np.log(df['price'] / df['price'].shift(1))

        # Return's lags
        for lag in range(1, self.n_lags + 1):
            df[f'return_lag_{lag}'] = df['return'].shift(lag)

        # Rolling stats on returns
        for w in self.windows:
            df[f'return_rolling_mean_{w}'] = df['return'].shift(1).rolling(w).mean()
            df[f'return_rolling_std_{w}'] =  df['return'].shift(1).rolling(w).std()

            
        
        # Returns
        #returns = df['price'].pct_change()
        #df['return_1d'] = returns
        #df['return_7d'] = returns.rolling(7).sum()
        ##df['momentum'] = df['price'] / df['price'].shift(7) - 1
#
        ###Volatility
        #df['volatility_7d'] = returns.rolling(7).std()
#
        ## RSI (14 days)
        #delta = df['price'].diff()
        #gain = delta.clip(lower=0).rolling(14).mean()
        #loss = (-delta.clip(upper=0)).rolling(14).mean()
        #rs = gain / loss
        #df['rsi'] = 100 - (100 / (1 + rs))
        ## MACD (12,26,9)
        #ema12 = df['price'].ewm(span=12, adjust=False).mean()
        #ema26 = df['price'].ewm(span=26, adjust=False).mean()
        #df['macd'] = ema12 - ema26
        #df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        #df['macd_hist'] = df['macd'] - df['macd_signal']

        self.logger.info(f"NaN count por columna antes de dropna:\n{df.isna().sum()}")

        df.dropna(inplace=True)
        return df


    def prepare_training_data(self, df: pd.DataFrame, n_horizon: int = 7) -> tuple:
        prices = df['price_usd'].values
        volumes = df['volume_usd'].values if 'volume_usd' in df.columns else None
        if volumes is not None:
            volumes =pd.to_numeric(volumes, errors='coerce')
            self.logger.info(f"Volumes - NaN count: {np.isnan(volumes).sum()}, sample: {volumes[:5]}")

        #DEBUGG
        self.logger.info(f"Input prices shape: {len(prices)}, NaN count: {np.isnan(prices).sum()}")

        min_required = self.n_lags + 1
        if len(prices) < min_required:
            raise ValueError(f"Not enough data: need at least {min_required} prices, got {len(prices)}")

        # Create  features
        feature_df = self._create_features(prices,volumes)

        if self.target_type == 'price':
            y = feature_df['price'].values
            exclude = ['price']
        elif self.target_type == 'return':
            if 'return' not in feature_df.columns:
                            raise ValueError("_create_features must generate 'return' column for target_type='return'")
            y = feature_df['return'].values
            exclude = ['price', 'return']
        elif self.target_type == 'diff':
        # Diferencia simple a 1 día (price_t - price_{t-1})
            if 'diff' not in feature_df.columns:
                feature_df['diff'] = feature_df['price'].diff()
            y = feature_df['diff'].values
            exclude = ['price', 'diff']
        else:
            raise ValueError(f"Unknown target_type: {self.target_type}")
       
        # Las features son todas las columnas excepto las excluidas
        self.feature_names = [col for col in feature_df.columns if col not in exclude]
        X = feature_df[self.feature_names].values

           # Verificar que X e y tengan la misma longitud
        if len(X) != len(y):
            raise ValueError(f"Length mismatch: X={len(X)}, y={len(y)}")


        self.last_prices = prices[-self.n_lags:].copy()
        self.logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
        return X, y, self.last_prices

        #if len(feature_df) == 0:
        #    self.logger.error(f"Feature creation failed. Input prices length: {len(prices)}")
        #    self.logger.error(f"First 10 prices: {prices[:10]}")
        #    self.logger.error(f"NaN count in prices: {np.isnan(prices).sum()}")
        #    raise ValueError(f"Feature creation failed: need at least {min_required} prices, got {len(prices)}")
#
        #self.feature_names = [col for col in feature_df.columns if col != 'price']
        ##X = feature_df.drop('price', axis=1).values
        ##y = feature_df['price'].values
#
        #if self.target_type == 'return':
        #    # Calcular retorno logarítmico
        #    feature_df['target'] = np.log(feature_df['price'] / feature_df['price'].shift(1))
        #    # Eliminar la primera fila (que tiene NaN por el shift)
        #    feature_df.dropna(subset=['target'], inplace=True)
        #    y = feature_df['target'].values
        #    # Las features se quedan igual (rezagos del precio, medias móviles, etc.)
        #    X = feature_df.drop(['price', 'target'], axis=1).values
        #else:
        #    y = feature_df['price'].values
        #    X = feature_df.drop('price', axis=1).values
       #
        #self.logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
        #return X, y, last_prices
#
#
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

