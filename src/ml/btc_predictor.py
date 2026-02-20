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
        #self.last_prices = None
        self.logger = logging.getLogger(__name__)

    def _create_features(self, prices: np.ndarray) -> pd.DataFrame:
        df = pd.DataFrame({'price': prices})
        for lag in range(1, self.n_lags + 1):
            df[f'lag_{lag}'] = df['price'].shift(lag)
        for w in self.windows:
            df[f'rolling_mean_{w}'] = df['price'].rolling(window=w, min_periods=1).mean()
            df[f'rolling_std_{w}'] = df['price'].rolling(window=w, min_periods=2).std()
        df.dropna(inplace=True)
        return df

    def prepare_training_data(self, df: pd.DataFrame) -> tuple:
        prices = df['price_usd'].values
        if len(prices) < self.n_lags:
            raise ValueError(f"Not enough data: need at least {self.n_lags} prices, got {len(prices)}")

        # Guardar los últimos precios para devolverlos (ya no como atributo)
        last_prices = prices[-self.n_lags:].copy()
        self.last_prices = last_prices
        self.logger.info(f"✅ last_prices (últimos 3): {last_prices[-3:]}")

        feature_df = self._create_features(prices)
        self.feature_names = [col for col in feature_df.columns if col != 'price']
        X = feature_df.drop('price', axis=1).values
        y = feature_df['price'].values

        self.logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
        return X, y, last_prices



    #def prepare_training_data(self, df: pd.DataFrame) -> tuple:
    #    prices = df['price_usd'].values
    #    if len(prices) < self.n_lags:
    #        raise ValueError(f"Not enough data: need at least {self.n_lags} prices, got {len(prices)}")
#
    #    # ASIGNACIÓN CLAVE
    #    self.last_prices = prices[-self.n_lags:].copy()
    #    self.logger.info(f"✅ last_prices GUARDADOS (últimos 3): {self.last_prices[-3:]}")
#
    #    feature_df = self._create_features(prices)
    #    self.feature_names = [col for col in feature_df.columns if col != 'price']
    #    X = feature_df.drop('price', axis=1).values
    #    y = feature_df['price'].values
    #    self.logger.info(f"Prepared {len(X)} samples with {X.shape[1]} features")
    #    return X, y

    def train(self, X, y):
        """
        Train the model.

        Args:
            X: Feature matrix (numpy array or DataFrame)
            y: Target values
        """
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


    #def train(self, X: np.ndarray, y: np.ndarray) -> None:
    #    
    #    X_scaled = self.scaler.fit_transform(X)
    #    self.model.fit(X_scaled, y)
    #    self.is_trained = True
#
    #    n_features = X.shape[1]
    #    self.feature_names = [f'feature_{i}' for i in range(n_features)]
#
    #    score = self.model.score(X_scaled, y)
    #    self.logger.info(f"Model trained. R² = {score:.4f}")


    def predict_future(self, n_days: int, last_prices=None) -> np.ndarray:
    
        if last_prices is not None:
            prices_to_use = last_prices
            self.logger.info(f"📦 Using provided last_prices (last 3): {prices_to_use[-3:]}")
        else:
            if not hasattr(self, 'last_prices') or self.last_prices is None:
                raise ValueError("No last_prices available")
            prices_to_use = self.last_prices

        predictions = []
        current_prices = list(prices_to_use)

        for step in range(n_days):
            # Crear Series de precios para usar _create_features
            prices_series = pd.Series(current_prices)

            # Usar el MISMO método que en training
            feature_df = self._create_features(prices_series.values)

            # Tomar la última fila (que corresponde al último precio)
            if len(feature_df) == 0:
                self.logger.error(f"❌ _create_features returned empty DataFrame at step {step}")
                raise ValueError("Cannot create features for prediction")

            # Obtener features de la última fila (sin la columna 'price')
            feature_row = feature_df.drop('price', axis=1).iloc[-1:].values

            # Verificar dimensiones
            if feature_row.shape[1] != len(self.feature_names):
                self.logger.error(f"❌ Feature mismatch: got {feature_row.shape[1]}, expected {len(self.feature_names)}")
                self.logger.error(f"   Created columns: {feature_df.drop('price', axis=1).columns.tolist()}")
                self.logger.error(f"   Expected: {self.feature_names}")
                raise ValueError(f"Feature count mismatch")

            # Escalar y predecir
            feature_vector_scaled = self.scaler.transform(feature_row)
            next_price = self.model.predict(feature_vector_scaled)[0]

            predictions.append(next_price)
            current_prices.append(next_price)

        return np.array(predictions)

    #def predict_future(self, n_days: int) -> np.ndarray:
    #    self.logger.info(f"🔍 predict_future: last_prices es {self.last_prices}")
    #    if self.last_prices is None:
    #        raise ValueError("No last prices available. Call prepare_training_data first.")
#
    #    history = list(self.last_prices)
    #    predictions = []
#
    #    for step in range(n_days):
    #        temp_prices = np.array(history + predictions)
    #        feature_df = self._create_features(temp_prices)
#
    #        if feature_df.empty:
    #            recent = temp_prices[-self.n_lags:]
    #            feature_vector = recent[::-1]
    #            self.logger.warning(f"Feature df empty at step {step}, using manual lags")
    #        else:
    #            feature_vector = feature_df.iloc[-1][self.feature_names].values
#
    #        feature_vector_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
    #        pred = self.model.predict(feature_vector_scaled)[0]
    #        predictions.append(pred)
#
    #    return np.array(predictions)
# =======================================================ORIGINAL===============================================================
#import numpy as np
#import pandas as pd
#import logging
#from sklearn.linear_model import LinearRegression
#from typing import Tuple, Optional, List
#
#
#class BTCPredictor:
#
#    def __init__(self, model: Optional[LinearRegression] = None):
#
#        self.model = model or LinearRegression()
#        self.logger = logging.getLogger(__name__)
#        self.is_trained = False 
#        self.training_size = 0 
#        #self.predict = predict
#    
#
#    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
#
#        prices = df["price_usd"].values
#        X = np.arange(len(prices)).reshape(-1, 1)
#        y = prices
#
#        self.training_size = len(X)
#        self.logger.info(f"Prepared data: {self.training_size} samples")
#        return X, y
#
#
#    def train(self, X: np.ndarray, y: np.ndarray) -> None:
#
#        self.logger.info(f" Training model with {len(X)} datos...")
#        self.model.fit(X, y)
#        self.is_trained = True
#
#        # “Returns a number that measures how well the model predicts.”
#        score = self.model.score(X, y)
#        self.logger.info(f"Trained model. R² score: {score:.4f}")
#
#
#    def predict_future(self, n_days:int) -> np.ndarray:
#        x_future = np.arange(
#            self.training_size,
#            self.training_size + n_days
#        ).reshape(-1,1)
#        predictions = self.model.predict(x_future)
#        self.logger.info(f"{n_days} predicciones generadas")
#        return predictions
#        
#        
#