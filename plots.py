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

    def _plot_prediction_with_metrics(
        self,
        df: pd.DataFrame,
        model_data: Dict,
        color: str,
        filename: str,
        title_suffix: str = ""
    ) -> str:
        """
        Generate standardized plot: historical data + predictions + metrics.
        
        Args:
            df: DataFrame with historical data
            model_data: Dictionary returned by _train_and_predict()
            color: Color for prediction line
            filename: Output PNG filename
            title_suffix: Additional text for title
        
        Returns:
            Path to saved plot file
        """
        # Unpack model data
        y = model_data['y']
        y_pred_train = model_data['y_pred_train']
        predictions = model_data['predictions']
        r2 = model_data['r2_score']
        mae = model_data['mae']
        rmse = model_data['rmse']
        future_dates = model_data['future_dates']
        last_date = model_data['last_date']
        last_price = model_data['last_price']
        model_name = model_data['model_name']

        std_error = np.std(y - y_pred_train)
        
        # Create figure with two subplots
        fig, (ax_main, ax_metrics) = plt.subplots(
            2, 1,
            figsize=(14, 10),
            height_ratios=[3, 1],
            gridspec_kw={'hspace': 0.3}
        )
        
        # ---- MAIN PLOT: Historical + Predictions ----
        # Historical data
        ax_main.plot(
            df['date'], df['price_usd'],
            color=color, linewidth=1.5, linestyle='-', alpha=0.7,
            label=f'{model_name} Training Fit (R²={r2:.3f})', zorder=4
        )
        
        # Future predictions
        ax_main.plot(
            future_dates, predictions,
            color=color, linestyle='--', linewidth=2.5, marker='D', markersize=6,
            markerfacecolor='white',markeredgewidth=2, markeredgecolor=color,
                 label=f'{model_name} Future Prediction', zorder=6)
        
        
      # Intervalo de confianza (sombreado)
        ax_main.fill_between(future_dates,
                             predictions - 1.96 * std_error,
                             predictions + 1.96 * std_error,
                             color=color, alpha=0.15, label='95% Confidence Interval', zorder=2)    


        # Transition point
        ax_main.scatter(
            [last_date], [last_price],
            color='red', s=150, zorder=10,
            edgecolor='white', linewidth=2,
            label='Prediction Start', marker='*'
        )

        ax_main.axvline(x=last_date, color='gray', linestyle=':', linewidth=1.5, alpha=0.7, zorder=4)

        
        # Shaded prediction area
        ax_main.axvspan(
            last_date, future_dates[-1],
            alpha=0.08, color='yellow', zorder=1,
            label='Prediction Period'
        )
        
        # Main plot configuration
        ax_main.set_title(f'{model_name} vs Real BTC Price\n{len(predictions)}-Day Prediction{title_suffix}',
                      fontsize=14, fontweight='bold')
        ax_main.set_xlabel('Date')
        ax_main.set_ylabel('Price (USD)')
        ax_main.legend(loc='upper left', fontsize=9)
        ax_main.grid(True, alpha=0.3)
        ax_main.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        ax_main.tick_params(axis='x', rotation=90)

        
        # ---- METRICS PLOT: Error Distribution ----
        # Calculate training errors
        errors = y - y_pred_train
        ax_metrics.hist(errors, bins=30, alpha=0.7, color=color, edgecolor='black')
        ax_metrics.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax_metrics.set_title(f'Error Distribution (Training) - {model_name}', fontsize=12, fontweight='bold')
        ax_metrics.set_xlabel('Error (USD)')
        ax_metrics.set_ylabel('Frequency')
        ax_metrics.grid(True, alpha=0.3)

        # Stats box
        stats_text = (f'MAE: ${mae:,.2f}\n'
                  f'RMSE: ${rmse:,.2f}\n'
                  f'R²: {r2:.4f}\n'
                  f'Std Error: ${std_error:,.2f}')
        ax_metrics.text(0.95, 0.95, stats_text,
                    transform=ax_metrics.transAxes,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                    fontsize=9)
        
        # Save figure
        plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Plot saved: {filepath}")
        return filepath
