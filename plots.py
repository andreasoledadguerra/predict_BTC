# plot.py

"""
Visualization script for Bitcoin price prediction models.

This module will contain functions to:
- Load prediction model results
- Generate comparison charts
- Display BTC price forecasts
"""

# TODO: Implement visualization functions

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from typing import Optional
from sklearn.linear_model import Ridge
from datetime import timedelta
from src.api.coingecko_client import CoinGeckoClient
from src.ml.btc_predictor import BTCPredictor
from src.pipeline.btc_pipeline import BTCDataPipeline

class BTCPlotter:

    def __init__(self, df: pd.DataFrame, output_dir: str = "plots"):
        self.df = df
        self.btc_predictor = BTCPredictor
        self.pipeline = BTCDataPipeline
        self.output_dir = output_dir

        #create folder
        os.makedirs(self.output_dir, exist_ok=True)

        self.colors = {
            'real': '#2E86AB',      # Azul
            'linear': '#A23B72',    # Magenta
            'ridge': '#F18F01' 
        }

        

    def prepare_training_data(self):
        return self.btc_predictor.prepare_training_data()
    
    def train(self):
        return self.btc_predictor.train()
    
    
    def predict_future(self):
        return self.btc_predictor.predict_future()
    
    
    def plot_real_prices(self, df: pd.DataFrame, title: str = "Precios Reales BTC") -> str:
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Asegurar que date sea datetime
        dates = pd.to_datetime(df['date'])
        
        # Plot simple de línea
        ax.plot(dates, df['price_usd'], 
                color=self.colors['real'], 
                linewidth=2,
                label='Precio Real BTC')
        
        # Puntos en inicio y fin
        ax.plot(dates.iloc[0], df['price_usd'].iloc[0], 'ko', markersize=8,
                label=f'Inicio: ${df["price_usd"].iloc[0]:.2f}')
        ax.plot(dates.iloc[-1], df['price_usd'].iloc[-1], 'ro', markersize=8,
                label=f'Fin: ${df["price_usd"].iloc[-1]:.2f}')
        
        # Configuración
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Fecha')
        ax.set_ylabel('Precio (USD)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Formato de fechas
        fig.autofmt_xdate()
        
        # Guardar
        filename = os.path.join(self.output_dir, "btc_real_prices.png")
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.close(fig)
        
        return filename
    

    def plot_model_lr(self, df: pd.DataFrame, n_days_future: int):

        # Train model
        linear_predictor = BTCPredictor()
        X,y = linear_predictor.prepare_training_data(df)
        linear_predictor.train(X, y)
        linear_pred = linear_predictor.predict_future(n_days_future)

        #Calculate R²
        r2_score = linear_predictor.model.score(X, y)

        # Combine historical + prections
        all_predictions = np.concatenate([y, linear_pred])

        # Generate future dates
        last_date = df['date'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=n_days_future,
            freq='D'
        )

        # Create visualization
        fig,(ax_main, ax_metrics) = plt.subplots(
            2, 1,
            figsize=(14, 10),
            height_ratios=[3, 1],
            gridspec_kw={'hspace': 0.3}
        )

        # Main plot
        ax_main.plot(df['date'], df['price_usd'], 
                'bo-', linewidth=2, markersize=4, alpha=0.7,
                label='Precio Real BTC')
        
        # Predicciones futuras
        ax_main.plot(future_dates, linear_pred, 
                    'r--', linewidth=2.5, markersize=5,
                    label=f'Predicción Linear (R²={r2_score:.3f})')
        
        # Punto de transición
        ax_main.scatter([last_date], [df['price_usd'].iloc[-1]], 
                       color='red', s=100, zorder=5,
                       edgecolor='black', linewidth=1.5,
                       label='Inicio Predicción')
        
        # Configuración del gráfico principal
        ax_main.set_title(
            f'Predicción de Precio BTC - {n_days_future} días futuros\n'
            f'Regresión Lineal Simple (R² = {r2_score:.4f})',
            fontsize=14, fontweight='bold', pad=15
        )
        ax_main.set_xlabel('Fecha')
        ax_main.set_ylabel('Precio (USD)', fontsize=11)
        ax_main.legend(loc='upper left', fontsize=10)
        ax_main.grid(True, alpha=0.3)
        
        # Área sombreada para predicción
        ax_main.axvspan(last_date, future_dates[-1], 
                       alpha=0.1, color='gray',
                       label='Período de Predicción')
        
        # Formato de fechas
        ax_main.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        ax_main.tick_params(axis='x', rotation=45)
        
        # ---- GRÁFICO DE MÉTRICAS ----
        # Calcular errores
        errors = y - all_predictions[:len(y)]
        
        # Histograma de errores
        ax_metrics.hist(errors, bins=30, alpha=0.7, color='blue', edgecolor='black')
        ax_metrics.axvline(x=0, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        ax_metrics.set_title('Distribución de Errores (Período de Entrenamiento)', 
                            fontsize=12, fontweight='bold')
        ax_metrics.set_xlabel('Error (USD)')
        ax_metrics.set_ylabel('Frecuencia')
        ax_metrics.grid(True, alpha=0.3)
        
        # Añadir estadísticas
        error_stats = (f'MAE: ${np.mean(np.abs(errors)):.2f}\n'
                      f'RMSE: ${np.sqrt(np.mean(errors**2)):.2f}\n'
                      f'R²: {r2_score:.4f}')
        
        ax_metrics.text(0.95, 0.95, error_stats,
                       transform=ax_metrics.transAxes,
                       verticalalignment='top',
                       horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                       fontsize=9)


        plt.tight_layout()

        # Save the file
        filename = os.path.join(self.output_dir, "btc_linear_prediction.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close(fig)

        print(f"✅ Linear plot guardado: {filename}")  # Debug


        return filename



    def plot_model_ridge(self, df: pd.DataFrame, n_days_future: int, alpha: float = 1.0):
    
        # 2. Entrenar modelo Ridge
    
        ridge_predictor = BTCPredictor(model=Ridge(alpha=alpha))
        X, y = ridge_predictor.prepare_training_data(df)
        ridge_predictor.train(X, y)
        ridge_pred = ridge_predictor.predict_future(n_days_future)
    
        # 3. Calcular métricas
        ridge_score = ridge_predictor.model.score(X, y)
    
        # 4. Combinar histórico + predicciones
        all_predictions = np.concatenate([y, ridge_pred])
    
        # 5. Generar fechas futuras
        last_date = df['date'].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=n_days_future,
            freq='D'
        )
    
        # 6. Crear visualización (MISMA ESTRUCTURA QUE plot_model_lr)
        fig, (ax_main, ax_metrics) = plt.subplots(
            2, 1,
            figsize=(14, 10),
            height_ratios=[3, 1],
            gridspec_kw={'hspace': 0.3}
        )
    
        # ---- GRÁFICO PRINCIPAL ----
        # Datos históricos
        ax_main.plot(df['date'], df['price_usd'], 
                    'ko-', linewidth=2, markersize=4, alpha=0.7,
                    label='Precio Real BTC')
    
         # Predicciones futuras RIDGE
        ax_main.plot(future_dates, ridge_pred, 
                    'r--', linewidth=2.5, markersize=5,
                    label=f'Ridge Regression (α={alpha}, R²={ridge_score:.3f})')
    
        # Punto de transición
        ax_main.scatter([last_date], [df['price_usd'].iloc[-1]], 
                    color='red', s=100, zorder=5,
                    edgecolor='black', linewidth=1.5,
                     label='Inicio Predicción')
    
        # Configuración del gráfico principal
        ax_main.set_title(
            f'Predicción de Precio BTC - {n_days_future} días futuros\n'
            f'Ridge Regression con Regularización L2 (α={alpha})',
            fontsize=14, fontweight='bold', pad=15
        )
        ax_main.set_xlabel('Fecha')
        ax_main.set_ylabel('Precio (USD)', fontsize=11)
        ax_main.legend(loc='upper left', fontsize=10)
        ax_main.grid(True, alpha=0.3)
    
        # Área sombreada para predicción
        ax_main.axvspan(last_date, future_dates[-1], 
                    alpha=0.1, color='gray',
                    label='Período de Predicción')
    
        # Formato de fechas
        ax_main.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m-%d'))
        ax_main.tick_params(axis='x', rotation=45)
    
        # ---- GRÁFICO DE MÉTRICAS ----
        # Calcular errores (usando predicciones del período de entrenamiento)
        errors = y - all_predictions[:len(y)]
    
        # Histograma de errores
        ax_metrics.hist(errors, bins=30, alpha=0.7, color='red', edgecolor='black')
        ax_metrics.axvline(x=0, color='blue', linestyle='--', linewidth=1.5, alpha=0.7)
    
        ax_metrics.set_title('Distribución de Errores - Ridge Regression', 
                            fontsize=12, fontweight='bold')
        ax_metrics.set_xlabel('Error (USD)')
        ax_metrics.set_ylabel('Frecuencia')
        ax_metrics.grid(True, alpha=0.3)
    
        # Añadir estadísticas
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
    
        error_stats = (f'MAE: ${mae:.2f}\n'
                    f'RMSE: ${rmse:.2f}\n'
                    f'R²: {ridge_score:.4f}\n'
                    f'α (alpha): {alpha}')
    
        ax_metrics.text(0.95, 0.95, error_stats,
                    transform=ax_metrics.transAxes,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=9)
    
        # 7. Ajustar layout
        plt.tight_layout()
    
        print(f"✅ Modelo Ridge entrenado: R² = {ridge_score:.4f}, α = {alpha}")
        print(f"📊 Predicciones: {n_days_future} días desde {future_dates[0].date()}")
    
        filename = os.path.join(self.output_dir, "btc_ridge_prediction.png")
        plt.savefig(filename, dpi= 150, bbox_inches='tight')
        plt.close(fig)

        print(f"✅ Ridge plot guardado: {filename}")  # Debug


        return filename

    def plot_all(self,
                df_real: pd.DataFrame,
                n_days_future: int = 10) -> dict:

        plot_paths = {}

        try:
            plot_paths['real'] = self.plot_real_prices(df_real)
        except Exception as e:
            print(f"❌ Error en plot_real_prices: {e}")
        
        try:
            plot_paths['linear'] = self.plot_model_lr(df_real, n_days_future)
        except Exception as e:
            print(f"❌ Error en plot_model_lr: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            plot_paths['ridge'] = self.plot_model_ridge(df_real, n_days_future)
        except Exception as e:
            print(f"❌ Error en plot_model_ridge: {e}")
            import traceback
            traceback.print_exc()
        
        return plot_paths
