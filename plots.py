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
from src.ml.optimization_ridge import optimize_ridge_alpha
from src.pipeline.btc_pipeline import BTCDataPipeline


logger = logging.getLogger(__name__)


class BTCPlotter:
    """
    Bitcoin price plotter with ML model predictions.
    
    Generates individual plots for Linear and Ridge regression models,
    plus a comparison plot showing both models together.
    """

    def __init__(self, df: pd.DataFrame, output_dir: str = "plots", n_lags: int =7, windows: List[int] = None):
        """
        Initialize the plotter.
        
        Args:
            df: DataFrame with historical BTC data
            output_dir: Directory to save plot images
        """
        self.df = df
        self.output_dir = output_dir 
        self.n_lags = n_lags
        self.windows = windows or [7, 14]
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

        # Store last trained models
        self.last_linear = None
        self.last_ridge = None

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


        # TRAINING DATA
        df = df.sort_values('date').reset_index(drop=True)
        df_train = df.iloc[:-n_days_future].copy() #train data - last n_days_future
        df_val = df.iloc[-n_days_future:].copy() # ground truth - last n_days_future

        X, y, _ = predictor.prepare_training_data(df_train)

        # Extended history for the first prediction (needed for rolling windows)
        history_size =min(200, len(df_train))
        extended_history = df_train['price_usd'].values[-history_size:].copy()

        ##
        #extended_prices = df['price_usd'].values[-20]

        # Train
        predictor.train(X, y)

        # Model trainig predictions (for plotting the fit)
        X_scaled = predictor.scaler.transform(X)
        y_pred_train = predictor.model.predict(X_scaled)

        # Metrics in train
        r2_train = predictor.model.score(X_scaled, y)
        mae_train = np.mean(np.abs(y - y_pred_train))
        rmse_train = np.sqrt(np.mean((y - y_pred_train) ** 2))

        # Predictions about the validation period (future)
        predictions_val = predictor.predict_future(n_days_future, last_prices=extended_history)

        # Metrics about validation ----------------------------------
        y_val = df_val['price_usd'].values[:len(predictions_val)]
        mae_val = np.mean(np.abs(y_val - predictions_val))
        rmse_val = np.sqrt(np.mean((y_val - predictions_val) ** 2))
        ss_res = np.sum((y_val - predictions_val) ** 2)
        ss_tot = np.sum((y_val - np.mean(y_val)) ** 2)
        r2_val = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        # ------------------------------------------------------------


        # Generate future dates for the plot
        last_date = df_train['date'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=n_days_future,
            freq='D'
        )


        # LOGS para comparar coeficientes (solo si el modelo tiene coef_)
        if hasattr(predictor.model, 'coef_'):
            logger.info(f"📊 {model_name} - Coeficientes (primeros 5): {predictor.model.coef_[:5]}")
            if len(predictor.model.coef_) > 5:
                logger.info("   ... (más coeficientes)")


        return {
            'X': X,
            'y': y,
            'y_pred_train': y_pred_train,
            'predictions_val': predictions_val,
            'r2_train': r2_train,
            'mae_train': mae_train ,
            'rmse_train': rmse_train,
            'r2_val': r2_val,
            'mae_val': mae_val,
            'rmse_val': rmse_val,
            'future_dates': future_dates,
            'model_name': model_name,
            'last_date': last_date,
            'last_price': df_train['price_usd'].iloc[-1],
            'predictor': predictor,
            'coef': predictor.model.coef_ if hasattr(predictor.model, 'coef_') else None,
            'intercept': predictor.model.intercept_ if hasattr(predictor.model, 'intercept_') else None,
            'df_val': df_val
        }


    def _plot_prediction_with_metrics(
        self,
        df: pd.DataFrame,
        model_data: Dict,
        color: str,
        filename: str,
        title_suffix: str = ""
    ) -> str:
   
        # Unpack model data
        y = model_data['y']
        y_pred_train = model_data['y_pred_train']
        predictions_val = model_data['predictions_val']
        r2_train = model_data['r2_train']
        mae_train = model_data['mae_train']
        rmse_train = model_data['rmse_train']
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
        df_val = model_data.get('df_val')

        # Historical data
        ax_main.plot(
            df['date'], df['price_usd'],
            color='#2E86AB', linewidth=2.5, 
            label='Real BTC Price', zorder=5, alpha=0.8
)
        # Validation

        if df_val is not None and not df_val.empty:
            # Get the last historical point
            last_train_date = df.iloc[:-len(df_val)]['date'].iloc[-1]
            last_train_price = df.iloc[:-len(df_val)]['price_usd'].iloc[-1]

            # Concatenate last point with initial one of df_val
            bridge_point = pd.DataFrame({
                'date': [last_train_date],
                'price_usd': [last_train_price]
            })

            df_val_connected = pd.concat([bridge_point, df_val]).reset_index(drop=True)
            ax_main.plot(
                df_val_connected['date'], df_val_connected['price_usd'],
                color='#2E86AB',        
                linewidth=2.5,
                linestyle='-',
                alpha=0.9,
                label='Real BTC Price (Validation)',
                zorder=7
    )

        # Adjust training model
        start_idx = len(df) - len(model_data['y_pred_train'])
        dates_train = df['date'].iloc[start_idx:].values
        ax_main.plot(
            dates_train, model_data['y_pred_train'],
            color=color, linewidth=2.0, linestyle='-', alpha=0.6,
            label=f'{model_name} Training Fit (R²={r2_train:.3f})', zorder=4
    )
        
        # Future predictions
        ax_main.plot(
            future_dates, predictions_val,
            color='red', linestyle='--', linewidth=2,
                 label=f'{model_name} Future Prediction', zorder=6)
        
        n_future = len(predictions_val)
        time_factor = np.sqrt(np.arange(1, n_future + 1))

        ci_lower = predictions_val - 1.96 * std_error * time_factor
        ci_upper = predictions_val + 1.96 * std_error * time_factor
        
        ax_main.fill_between(future_dates,
                             ci_lower,
                             ci_upper,
                             color=color, alpha=0.15, label='95% Confidence Interval (growing)', zorder=2)    


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
        ax_main.set_title(f'{model_name} vs Real BTC Price\n{len(predictions_val)}-Day Prediction{title_suffix}',
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
        std_error = np.std(errors)
        standardized_errors = errors / std_error #z-scores

        ax_metrics.hist(standardized_errors, bins=30, alpha=0.7, color=color, edgecolor='black',density=True)
        ax_metrics.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
        ax_metrics.axvline(x=-1.96 , color='red' , linestyle=':', linewidth=1, alpha=0.7, label='±1.96σ')
        ax_metrics.axvline(x= 1.96, color= 'red', linestyle=':', linewidth=1, alpha=0.7)
        ax_metrics.set_title(f'Standardized Residuals - {model_name}')
        ax_metrics.set_xlabel('Standardized Error (z-score)')
        ax_metrics.set_ylabel('Density')
        ax_metrics.grid(True, alpha=0.3)

        # Stats box
        stats_text = (f'TRAIN:\n'
                      f'MAE: ${model_data["mae_train"]:,.2f}\n'
                      f'RMSE: ${model_data["rmse_train"]:,.2f}\n'
                      f'R²: {model_data["r2_train"]:.4f}\n'
                      f'Std Error: ${std_error:,.2f}')

        if 'mae_val' in model_data and model_data['mae_val'] is not None:
            stats_text += (f'\n\nVALIDATION:\n'
                           f'MAE: ${model_data["mae_val"]:,.2f}\n'
                           f'RMSE: ${model_data["rmse_val"]:,.2f}\n'
                           f'R²: {model_data["r2_val"]:.4f}')

        ax_metrics.text(0.95, 0.95, stats_text,
                        transform=ax_metrics.transAxes,
                        verticalalignment='top', horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        fontsize=9)

        
        # Save figure
        #plt.tight_layout()
        filepath = os.path.join(self.output_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Plot saved: {filepath}")
        return filepath

    def plot_model_lr(
            self, 
            df_train: pd.DataFrame, 
            df_val: Optional[pd.DataFrame] = None,
            n_days_future: int = 3
        ) -> str:
    
            logger.info(f"\n📊 Training Linear Regression model...")
            model_data = self._train_and_predict(df_train, 'linear', n_days_future, df_val= df_val)
            self.last_linear = model_data
            
            return self._plot_prediction_with_metrics(
                df=df_train,
                df_val=df_val,
                model_data=model_data,
                color=self.colors['linear'],
                filename="btc_linear_comparison.png"
            )
    
    def plot_model_ridge(
            self, 
            df_train: pd.DataFrame,
            df_val: Optional[pd.DataFrame] = None, 
            n_days_future: int = 3,
            alpha: float = 1.0,
            optimize_alpha: bool = True
        ) -> str:
            """
            Generate plot for Ridge Regression model.
            
            Creates individual plot: "Real Price vs Ridge Regression Prediction"
            
            Args:
                df: Historical data DataFrame
                n_days_future: Number of days to predict
                alpha: Regularization parameter (L2 penalty)
            
            Returns:
                Path to saved PNG file
            """
            logger.info(f"\n📊 Training Ridge Regression model (α={alpha})...")

            #Optional optimization
            if optimize_alpha:
                temp_predictor = BTCPredictor(n_lags=self.n_lags, windows=self.windows)
                X_all, y_all, _ = temp_predictor.prepare_training_data(df_train)
                best_alpha, best_score = optimize_ridge_alpha(X_all, y_all)

                logger.info(f"Optimization complete: best alpha = {best_alpha} (CV R² = {best_score:.4f})")
                alpha = best_alpha



            model_data = self._train_and_predict(df_train, 
                'ridge', 
                n_days_future, 
                alpha=alpha, 
                df_val=df_val)
            self.last_ridge = model_data
            suffix= f" | α={alpha} "
            
            return self._plot_prediction_with_metrics(
                df=df_train,
                df_val=df_val,
                model_data=model_data,
                color=self.colors['ridge'],
                filename="btc_ridge_comparison.png",
                title_suffix=suffix
            )
    
    
    def plot_all(
        self,
        df_real: pd.DataFrame,
        n_days_future: int = 3,
        alpha: float = 1.0,
        optimize_alpha: bool = True
    ) -> Dict[str, str]:
        """
        Generate all plots: linear model, ridge model, and comparison.
        
        Args:
            df_real: Historical data DataFrame
            n_days_future: Number of days to predict
            alpha: Ridge regularization parameter
        
        Returns:
            Dictionary mapping plot types to file paths:
                - 'linear': Linear regression plot
                - 'ridge': Ridge regression plot
                - 'comparison': Comparison plot
        """

        logger.info("\n" + "="*60)
        logger.info("🚀 GENERATING ALL BTC PREDICTION PLOTS")
        logger.info("="*60)
        
        # Sort and divide chronologically
        df_real = df_real.sort_values('date').reset_index(drop=True)
        n = len(df_real)
        split_idx = int(n * 0.8)  # 80% train, 20% validation
        df_train = df_real.iloc[:split_idx].copy()
        df_val = df_real.iloc[split_idx:].copy()

        logger.info(f"Split: {len(df_train)} train, {len(df_val)} validation samples")


        result = {
             'paths': {},
             'linear_model': None,
             'ridge_model' : None
        }
        
        
        # Plot 1: Linear Regression
        try:
            logger.info("\n[1/2] Plotting Linear Regression model...")
            result['paths']['linear'] = self.plot_model_lr(
                df_train=df_train,
                df_val=df_val,
                n_days_future=n_days_future
                )
            result['linear_model'] = self.last_linear
        except Exception as e:
            print(f"❌ Error in plot_model_lr: {e}")
            import traceback
            traceback.print_exc()
        
        # Plot 2: Ridge model
        try:
            logger.info("\n[2/2] Plotting Ridge Regression model...")
            result['paths']['ridge'] = self.plot_model_ridge(
                df_train=df_train,
                df_val=df_val, 
                n_days_future=n_days_future,
                alpha=alpha,
                optimize_alpha=optimize_alpha
                )
            result['ridge_model'] = self.last_ridge
        except Exception as e:
            print(f"❌ Error in plot_model_ridge: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("\n" + "="*60)
        logger.info(f"✅ COMPLETED: {len(result['paths'])}/2 plots generated successfully")
        logger.info("="*60)
        
        return result
