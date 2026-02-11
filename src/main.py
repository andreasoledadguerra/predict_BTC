import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

from src.config.settings import get_postgres_settings, get_coingecko_settings
from src.api.coingecko_client import CoinGeckoClient
from src.database.postgres_manager import DatabaseManager
from src.ml.btc_predictor import BTCPredictor
from src.pipeline.btc_pipeline import BTCDataPipeline
from src.config.logs import configure_logging
from plots import BTCPlotter

logger = logging.getLogger(__name__)

load_dotenv()
configure_logging()
# ====================================== SETTING =================================================
def initialize_components():
    pg_settings = get_postgres_settings()
    cg_settings = get_coingecko_settings()

    api_client =CoinGeckoClient(API_KEY=cg_settings.API_KEY, timeout=10 )

    db_manager = DatabaseManager(
      settings= pg_settings
    )

    predictor = BTCPredictor()

    pipeline = BTCDataPipeline(api_client,db_manager, predictor)

    return pipeline, db_manager


# ====================================== STAGE 1 ================================================
def run_stage1_fetch(pipeline: BTCDataPipeline):
    """
    Execute STAGE 1: Get data from CoinGecko and save to DB.
    """
    logger.info("\n" + "=" * 60)
    logger.info(" STAGE 1: FETCH DATA FROM COINGECKO")
    logger.info("=" * 60)
    
    logger.info("\n INSTRUCTIONS:")
    logger.info("   - Maximum 90 days for CoinGecko free API")
    logger.info("   - Data will be saved to PostgreSQL")
    logger.info("   - Run this FIRST before making predictions")
    
    logger.info("\n Enter date range to DOWNLOAD:")

    fetch_start = input(" Start date (YYYY-MM-DD): ").strip()
    fetch_end = input(" End date (YYYY-MM-DD): ").strip()

    result_fetch = pipeline.fetch_data(fetch_start, fetch_end)

    result_save =pipeline.save_data_in_db(result_fetch)


    logger.info("\n" + "-" * 40)
    logger.info(" STAGE 1 RESULTS")
    logger.info("-" * 40)

    logger.info(f" SUCCESS: {result_save} records saved")

# ========================================= STAGE 2 =====================================================
def run_stage2_train_predict(pipeline: BTCDataPipeline):
    """
    Execute STAGE 2: Get data from DB, train model, predict and show plots
    """
    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2: TRAIN MODEL, PREDICT AND PLOT")
    logger.info("=" * 60)

    logger.info("\n Enter date range to TRAIN the model:")
    logger.info("Must be within the available range")

    train_start = input("Start date (YYYY-MM-DD): ").strip()
    train_end = input("End date (YYYY-MM-DD): ").strip()

    logger.info(f"\n Checking data for {train_start} to {train_end}")
    has_data = pipeline.get_data_for_training(train_start, train_end)

    days_input = input("\n How many days do you want to predict? (10): ").strip()
    predict_days = int(days_input) if days_input else 10

    logger.info(f"\n Training model with data from {train_start} to {train_end}")
    logger.info(f"Predicting {predict_days} days into the future...")

    # Obtener predicciones
    predictions = pipeline.predict_training_data(has_data, train_start, train_end, predict_days)

    # ======================================= GENERATE PLOTS ==========================================
    logger.info("\n📊 Generando visualizaciones...")
    
    try:
        # Importar plotter
        from plots import BTCPlotter
        
        # Inicializar plotter
        plotter = BTCPlotter(df=has_data, output_dir="plots")
        
        # Asegurar que predictions sea un diccionario
        if not isinstance(predictions, dict):
            logger.warning("Las predicciones no son un diccionario, creando estructura...")
            # Crear estructura básica
            predictions = {
                'linear': predictions,
                'ridge': predictions,  # Mismo para demostración
                'linear_r2': 0.0,
                'ridge_r2': 0.0
            }
        
        # Generar los tres plots
        plot_paths = plotter.plot_all(
            df_real=has_data,  # DataFrame con precios reales
            n_days_future=predict_days
        )
        
        logger.info("\n✅ Gráficos generados:")
        for model, path in plot_paths.items():
            logger.info(f"   Model: {model}. Path: {path}")
            
    except ImportError as e:
        logger.warning(f"\n⚠️  No se pudieron generar plots: {e}")
        logger.info("   Asegúrate de tener matplotlib instalado: pip install matplotlib")
    except Exception as e:
        logger.error(f"\n❌ Error generando plots: {e}")
        import traceback
        logger.error(f"Detalles: {traceback.format_exc()}")

    logger.info("\n" + "=" * 60)
    logger.info("STAGE 2 RESULTS")
    logger.info("=" * 60)

    logger.info(f"SUCCESS: Model trained and {predict_days} predictions generated")

    logger.info(f"\n📈 PREDICTIONS FOR THE NEXT {predict_days} DAYS:")

    # Mostrar predicciones (ajusta según tu estructura)
    if isinstance(predictions, dict):
        linear_preds = predictions.get("linear", [])
        ridge_preds = predictions.get("ridge", [])

        if linear_preds:
            logger.info("\n📊 Linear Regression:")
            # Mostrar solo las predicciones futuras
            future_start_idx = len(has_data) if len(linear_preds) > len(has_data) else 0
            for i, p in enumerate(linear_preds[future_start_idx:], start=1):
                logger.info(f"   Day {i}: ${p:.2f}")

        if ridge_preds:
            logger.info("\n🎯 Ridge Regression:")
            future_start_idx = len(has_data) if len(ridge_preds) > len(has_data) else 0
            for i, p in enumerate(ridge_preds[future_start_idx:], start=1):
                logger.info(f"   Day {i}: ${p:.2f}")
    else:
        # Fallback si predictions no es dict
        logger.info("\n📈 Predictions:")
        for i, p in enumerate(predictions, start=1):
            logger.info(f"   Day {i}: ${p:.2f}")

    return predictions



## ======================================= MAIN FUNCTION =================================================

def main():
    logger.info("=" * 60)
    logger.info("BITCOIN PRICE PREDICTOR")
    logger.info("=" * 60)

    # Initialize components
    pipeline, db_manager = initialize_components()

    while True:
        logger.info("\n" + "=" * 60)
        logger.info(" MAIN MENU")
        logger.info("=" * 60)
        logger.info("\nSelect an option:")
        logger.info("   1️⃣  STAGE 1: Fetch data from CoinGecko and save to DB")
        logger.info("   2️⃣  STAGE 2: Train model and predict (using DB data) and show plots")
        #logger.info("   3️⃣  STAGE 3: Plot real prices, linear regression model and ridge regression model")
        #logger.info("   4️⃣  EXIT")
        logger.info("   3️⃣  EXIT")

        choice = input("\n   Your choice (1-4): ").strip()

        if choice == "1":
              run_stage1_fetch(pipeline)
        elif choice == "2":
            run_stage2_train_predict(pipeline)
        #elif choice == "3":
        #    plot_all_models(pipeline)
        elif choice == "3":
            logger.info("\n👋 Thank you for using Bitcoin Price Predictor!")
            break
        else:
              logger.info("❌ Invalid option. Please choose 1-4.")
    
    if db_manager:
        db_manager.close()
        logger.info(" Database connection closed")

    logger.info("\n Program finished")

# ========================================= EXECUTION ===========================================

if __name__ == "__main__":
    main()