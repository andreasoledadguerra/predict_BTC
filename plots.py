# plot.py

# TODO: Implement visualization functions

"""
Visualization script for Bitcoin price prediction models.

This module contains functions to:
- Load prediction model results
- Generate comparison charts
- Display BTC price forecasts

#Refactored version with DRY principles and separation of concerns.
#"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging
from typing import Optional, Dict, List, Any
from sklearn.linear_model import Ridge
from datetime import timedelta
from src.api.coingecko_client import CoinGeckoClient
from src.ml.btc_predictor import BTCPredictor
from src.pipeline.btc_pipeline import BTCDataPipeline


logger = logging.getLogger(__name__)


class BTCPlotter:
    """
    Bitcoin price plotter with ML model predictions.
    
    Generates individual plots for Linear and Ridge regression models,
    plus a comparison plot showing both models together.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str = "plots"):
        """
        Initialize the plotter.
        
        Args:
            df: DataFrame with historical BTC data
            output_dir: Directory to save plot images
        """
        self.df = df
        self.btc_predictor = BTCPredictor
        self.pipeline = BTCDataPipeline
        self.output_dir = output_dir

        # Create output folder
        os.makedirs(self.output_dir, exist_ok=True)

        # Color scheme
        self.colors = {
            'real': '#2E86AB',      # Blue
            'linear': '#A23B72',    # Magenta
            'ridge': '#F18F01'      # Orange
        }

    # ============================================================
    # PRIVATE HELPER METHODS 
    # ============================================================

    def _train_and_predict(
        self, 
        df: pd.DataFrame, 
        model_type: str,
        n_days_future: int,
        **model_kwargs
    ) -> Dict:
        """
        Train a model and generate predictions.
        
        Args:
            df: Historical data DataFrame
            model_type: 'linear' or 'ridge'
            n_days_future: Number of days to predict
            **model_kwargs: Additional model parameters (e.g., alpha for Ridge)
        
        Returns:
            Dictionary containing:
                - X: Training features
                - y: Training targets
                - predictions: Future predictions
                - r2_score: Model R² score
                - future_dates: DatetimeIndex for predictions
                - model_name: Human-readable model name
                - last_date: Last date in historical data
                - predictor: Trained predictor instance
        """
        # Select and configure model
        if model_type == 'linear':
            predictor = BTCPredictor()
            model_name = "Linear Regression"
        elif model_type == 'ridge':
            alpha = model_kwargs.get('alpha', 1.0)
            predictor = BTCPredictor(model=Ridge(alpha=alpha))
            model_name = f"Ridge Regression (α={alpha})"
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        

        df = df.sort_values('date').reset_index(drop=True)

        # Preparing daa
        X, y = predictor.prepare_training_data(df)

        # Train
        predictor.train(X, y)

        # 
        y_pred_train = predictor.model.predict(X)

        # Metrics
        r2 = predictor.model.score(X, y)
        mae = np.mean(np.abs(y - y_pred_train))
        rmse = np.sqrt(np.mean(y - y_pred_train))


        predictions = predictor.predict_future(n_days_future)


        # Generate future dates
        last_date = df['date'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=n_days_future,
            freq='D'
        )
        
        last_price = df['price_usd'].iloc[-1]

        return {
            'X': X,
            'y': y,
            'y_pred_train': y_pred_train,
            'predictions': predictions,
            'r2_score': r2,
            'mae': mae ,
            'rmse': rmse,
            'future_dates': future_dates,
            'model_name': model_name,
            'last_date': last_date,
            'predictor': predictor
        }

    