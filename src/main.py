"""
Bitcoin Price Predictor - CLI Application

Main entry point for the BTC prediction system.
Provides interactive menu for data fetching, model training, and visualization.
"""

import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
from datetime import timedelta

from src.config.settings import get_postgres_settings, get_coingecko_settings
from src.api.coingecko_client import CoinGeckoClient
from src.database.postgres_manager import DatabaseManager
from src.ml.btc_predictor import BTCPredictor
from src.pipeline.btc_pipeline import BTCDataPipeline
from src.config.logs import configure_logging
from plots import BTCPlotter

logger = logging.getLogger(__name__)

# Load environment variables and configure logging
load_dotenv()
configure_logging()


# ====================================== INITIALIZATION =================================================

def initialize_components():
    """
    Initialize all system components.
    
    Returns:
        tuple: (pipeline, db_manager, plotter)
    """
    logger.info("🔧 Initializing components...")
    
    # Load settings
    pg_settings = get_postgres_settings()
    cg_settings = get_coingecko_settings()

    # Initialize API client
    api_client = CoinGeckoClient(
        API_KEY=cg_settings.API_KEY, 
        timeout=10
    )

    # Initialize database manager
    db_manager = DatabaseManager(settings=pg_settings)

    # Initialize ML predictor
    predictor = BTCPredictor()

    # Initialize data pipeline
    pipeline = BTCDataPipeline(api_client, db_manager, predictor)

    # Initialize plotter (always available)
    plotter = BTCPlotter(df=None, output_dir="plots")  # df will be set later

    logger.info("✅ All components initialized successfully")
    
    return pipeline, db_manager, plotter


# ====================================== STAGE 1: FETCH DATA ================================================

def run_stage1_fetch(pipeline: BTCDataPipeline):
    """
    Execute STAGE 1: Fetch data from CoinGecko API and save to PostgreSQL.
    
    Args:
        pipeline: BTCDataPipeline instance
    """
    logger.info("\n" + "=" * 60)
    logger.info("🌐 STAGE 1: FETCH DATA FROM COINGECKO")
    logger.info("=" * 60)
    
    logger.info("\n📋 INSTRUCTIONS:")
    logger.info("   • Maximum 90 days for CoinGecko free API")
    logger.info("   • Data will be saved to PostgreSQL")
    logger.info("   • Run this FIRST before making predictions")
    
    logger.info("\n📅 Enter date range to DOWNLOAD:")

    fetch_start = input("   Start date (YYYY-MM-DD): ").strip()
    fetch_end = input("   End date (YYYY-MM-DD): ").strip()

    try:
        # Validate dates
        start_dt = datetime.strptime(fetch_start, "%Y-%m-%d")
        end_dt = datetime.strptime(fetch_end, "%Y-%m-%d")
    except ValueError:
        logger.error("❌ Invalid date format. Please use YYYY-MM-DD")
        return

    #logger.info(f"\n🔄 Fetching data from {fetch_start} to {fetch_end}...")
    
    # Iterate day by day
    try:
        missing_dates = []
        current_dt = start_dt

        while current_dt <= end_dt:
            date_str = current_dt.strftime("%Y-%m-%d")
            day_data = db_manager.get_btc_prices(date_str, date_str)
        
        if day_data is not None and len(day_data) > 0:
            logger.debug(f"{date_str} already in DB, skipping.")
        else:
            logger.info(f" {date_str} not found in DB, will fetch.")
            missing_dates.append(date_str)
        
        current_dt += timedelta(days=1)




    #try:
#
    #    # verificar si los datos ya estaba en la base de datos    
    #    df = db_manager.get_btc_prices(fetch_start, fetch_end)
    #    if df != None :
    #        pass
    #    else:
    #        # Fetch data from API
    #        result_fetch = pipeline.fetch_data(fetch_start, fetch_end)                      
    #        if result_fetch is None or len(result_fetch) == 0:
    #            logger.warning("⚠️  No data fetched from CoinGecko")
            
        # Save to database
        result_save = pipeline.save_data_in_db(result_fetch)

        logger.info("\n" + "-" * 60)
        logger.info("📊 STAGE 1 RESULTS")
        logger.info("-" * 60)
        logger.info(f"✅ SUCCESS: {result_save} records saved to database")
        logger.info(f"📅 Date range: {fetch_start} to {fetch_end}")
        
    except Exception as e:
        logger.error(f"❌ Error in Stage 1: {e}")
        import traceback
        logger.debug(traceback.format_exc())


# ====================================== STAGE 2: TRAIN & PREDICT =====================================================

def run_stage2_train_predict(pipeline: BTCDataPipeline, plotter: BTCPlotter):
    """
    Execute STAGE 2: Load data from DB, train models, predict, and generate plots.
    
    Args:
        pipeline: BTCDataPipeline instance
        plotter: BTCPlotter instance
    """
    logger.info("\n" + "=" * 60)
    logger.info("🤖 STAGE 2: TRAIN MODEL, PREDICT & VISUALIZE")
    logger.info("=" * 60)

    # ---- GET DATE RANGE ----
    logger.info("\n📅 Enter date range to TRAIN the model:")
    logger.info("   (Must be within the available data in database)")

    train_start = input("   Start date (YYYY-MM-DD): ").strip()
    train_end = input("   End date (YYYY-MM-DD): ").strip()

    try:
        datetime.strptime(train_start, "%Y-%m-%d")
        datetime.strptime(train_end, "%Y-%m-%d")
    except ValueError:
        logger.error("❌ Invalid date format. Please use YYYY-MM-DD")
        return

    # ---- LOAD DATA FROM DB ----
    logger.info(f"\n🔍 Checking data availability for {train_start} to {train_end}...")
    
    try:
        df_train = pipeline.get_data_for_training(train_start, train_end)
        
        if df_train is None or len(df_train) == 0:
            logger.error("❌ No data found in database for the specified date range")
            logger.info("💡 Tip: Run STAGE 1 first to fetch data from CoinGecko")
            return
        
        logger.info(f"✅ Loaded {len(df_train)} records from database")
        
    except Exception as e:
        logger.error(f"❌ Error loading data from database: {e}")
        return

    # ---- GET PREDICTION DAYS ----
    days_input = input("\n🔮 How many days do you want to predict? (default: 10): ").strip()
    predict_days = int(days_input) if days_input and days_input.isdigit() else 10

    # ---- GET RIDGE ALPHA (OPTIONAL) ----
    alpha_input = input("🎛️  Ridge alpha (regularization parameter, default: 1.0): ").strip()
    alpha = float(alpha_input) if alpha_input else 1.0

    logger.info(f"\n📊 Training models with data from {train_start} to {train_end}")
    logger.info(f"🔮 Predicting {predict_days} days into the future")
    logger.info(f"🎯 Ridge alpha: {alpha}")


    logger.info("\n" + "-" * 60)
    logger.info("📊 GENERATING VISUALIZATIONS")
    logger.info("-" * 60)

    try:
        results = plotter.plot_all(
            df_real=df_train,
            n_days_future=predict_days,
            alpha=alpha
        )

        logger.info("\n✅ Plots generated successfully:")
        for model_type, path in results['paths'].items():
            logger.info(f"   📈 {model_type.upper()}: {path}")

    except Exception as e:
        logger.error(f"❌ Error generating plots: {e}")
        import traceback
        traceback.print_exc()
        return

    # ---- DISPLAY PREDICTIONS FROM MODEL DATA ----
    logger.info("\n" + "=" * 60)
    logger.info("📈 PREDICTION RESULTS")
    logger.info("=" * 60)

    linear_data = results.get('linear_model')
    ridge_data = results.get('ridge_model')

    if linear_data:
        logger.info(f"\n📊 Linear Regression (R² = {linear_data['r2_score']:.4f}):")
        for i, pred in enumerate(linear_data['predictions'], start=1):
            logger.info(f"   Day {i}: ${pred:,.2f}")

    if ridge_data:
        logger.info(f"\n🎯 Ridge Regression (R² = {ridge_data['r2_score']:.4f}, α = {alpha}):")
        for i, pred in enumerate(ridge_data['predictions'], start=1):
            logger.info(f"   Day {i}: ${pred:,.2f}")

    logger.info("\n" + "=" * 60)
    logger.info("✅ STAGE 2 COMPLETED SUCCESSFULLY")
    logger.info("=" * 60)


# ====================================== MAIN MENU =================================================

def display_menu():
    """Display the main menu options."""
    logger.info("\n" + "=" * 60)
    logger.info("📋 MAIN MENU")
    logger.info("=" * 60)
    logger.info("\nSelect an option:")
    logger.info("   1️⃣  STAGE 1: Fetch data from CoinGecko → Save to DB")
    logger.info("   2️⃣  STAGE 2: Train models → Predict → Generate plots")
    logger.info("   3️⃣  EXIT")


def main():
    """
    Main application entry point.
    
    Provides interactive CLI menu for:
    1. Fetching BTC data from CoinGecko
    2. Training ML models and making predictions
    3. Generating visualizations
    """
    logger.info("\n" + "=" * 60)
    logger.info("₿  BITCOIN PRICE PREDICTOR")
    logger.info("=" * 60)
    logger.info("\n🚀 Initializing application...")

    # Initialize all components
    try:
        pipeline, db_manager, plotter = initialize_components()
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {e}")
        logger.error("   Please check your configuration and try again")
        return

    # Main menu loop
    while True:
        display_menu()
        
        choice = input("\n   Your choice (1-3): ").strip()

        if choice == "1":
            run_stage1_fetch(pipeline)
            
        elif choice == "2":
            run_stage2_train_predict(pipeline, plotter)
            
        elif choice == "3":
            logger.info("\n" + "=" * 60)
            logger.info("👋 Thank you for using Bitcoin Price Predictor!")
            logger.info("=" * 60)
            break
            
        else:
            logger.warning("❌ Invalid option. Please choose 1-3.")
    
    # Cleanup
    if db_manager:
        db_manager.close()
        logger.info("🔌 Database connection closed")

    logger.info("\n✅ Program finished successfully\n")


# ====================================== EXECUTION ===========================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️  Program interrupted by user (Ctrl+C)")
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error(f"\n❌ Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)





































#import os
#import sys
#import logging
#from datetime import datetime
#from dotenv import load_dotenv
#
#from src.config.settings import get_postgres_settings, get_coingecko_settings
#from src.api.coingecko_client import CoinGeckoClient
#from src.database.postgres_manager import DatabaseManager
#from src.ml.btc_predictor import BTCPredictor
#from src.pipeline.btc_pipeline import BTCDataPipeline
#from src.config.logs import configure_logging
#from plots import BTCPlotter
#
#logger = logging.getLogger(__name__)
#
#load_dotenv()
#configure_logging()
## ====================================== SETTING =================================================
#def initialize_components():
#    pg_settings = get_postgres_settings()
#    cg_settings = get_coingecko_settings()
#
#    api_client =CoinGeckoClient(API_KEY=cg_settings.API_KEY, timeout=10 )
#
#    db_manager = DatabaseManager(
#      settings= pg_settings
#    )
#
#    predictor = BTCPredictor()
#
#    pipeline = BTCDataPipeline(api_client,db_manager, predictor)
#
#    return pipeline, db_manager
#
#
## ====================================== STAGE 1 ================================================
#def run_stage1_fetch(pipeline: BTCDataPipeline):
#    """
#    Execute STAGE 1: Get data from CoinGecko and save to DB.
#    """
#    logger.info("\n" + "=" * 60)
#    logger.info(" STAGE 1: FETCH DATA FROM COINGECKO")
#    logger.info("=" * 60)
#    
#    logger.info("\n INSTRUCTIONS:")
#    logger.info("   - Maximum 90 days for CoinGecko free API")
#    logger.info("   - Data will be saved to PostgreSQL")
#    logger.info("   - Run this FIRST before making predictions")
#    
#    logger.info("\n Enter date range to DOWNLOAD:")
#
#    fetch_start = input(" Start date (YYYY-MM-DD): ").strip()
#    fetch_end = input(" End date (YYYY-MM-DD): ").strip()
#
#    result_fetch = pipeline.fetch_data(fetch_start, fetch_end)
#
#    result_save =pipeline.save_data_in_db(result_fetch)
#
#
#    logger.info("\n" + "-" * 40)
#    logger.info(" STAGE 1 RESULTS")
#    logger.info("-" * 40)
#
#    logger.info(f" SUCCESS: {result_save} records saved")
#
## ========================================= STAGE 2 =====================================================
#def run_stage2_train_predict(pipeline: BTCDataPipeline):
#    """
#    Execute STAGE 2: Get data from DB, train model, predict and show plots
#    """
#    logger.info("\n" + "=" * 60)
#    logger.info("STAGE 2: TRAIN MODEL, PREDICT AND PLOT")
#    logger.info("=" * 60)
#
#    logger.info("\n Enter date range to TRAIN the model:")
#    logger.info("Must be within the available range")
#
#    train_start = input("Start date (YYYY-MM-DD): ").strip()
#    train_end = input("End date (YYYY-MM-DD): ").strip()
#
#    logger.info(f"\n Checking data for {train_start} to {train_end}")
#    has_data = pipeline.get_data_for_training(train_start, train_end)
#
#    days_input = input("\n How many days do you want to predict? (10): ").strip()
#    predict_days = int(days_input) if days_input else 10
#
#    logger.info(f"\n Training model with data from {train_start} to {train_end}")
#    logger.info(f"Predicting {predict_days} days into the future...")
#
#    # Obtener predicciones
#    predictions = pipeline.predict_training_data(has_data, train_start, train_end, predict_days)
#
#    # ======================================= GENERATE PLOTS ==========================================
#    logger.info("\n📊 Generando visualizaciones...")
#    
#    try:
#        # Importar plotter
#        from plots import BTCPlotter
#        
#        # Inicializar plotter
#        plotter = BTCPlotter(df=has_data, output_dir="plots")
#        
#        # Asegurar que predictions sea un diccionario
#        if not isinstance(predictions, dict):
#            logger.warning("Las predicciones no son un diccionario, creando estructura...")
#            # Crear estructura básica
#            predictions = {
#                'linear': predictions,
#                'ridge': predictions,  # Mismo para demostración
#                'linear_r2': 0.0,
#                'ridge_r2': 0.0
#            }
#        
#        # Generar los tres plots
#        plot_paths = plotter.plot_all(
#            df_real=has_data,  # DataFrame con precios reales
#            n_days_future=predict_days
#        )
#        
#        logger.info("\n✅ Gráficos generados:")
#        for model, path in plot_paths.items():
#            logger.info(f"   Model: {model}. Path: {path}")
#            
#    except ImportError as e:
#        logger.warning(f"\n⚠️  No se pudieron generar plots: {e}")
#        logger.info("   Asegúrate de tener matplotlib instalado: pip install matplotlib")
#    except Exception as e:
#        logger.error(f"\n❌ Error generando plots: {e}")
#        import traceback
#        logger.error(f"Detalles: {traceback.format_exc()}")
#
#    logger.info("\n" + "=" * 60)
#    logger.info("STAGE 2 RESULTS")
#    logger.info("=" * 60)
#
#    logger.info(f"SUCCESS: Model trained and {predict_days} predictions generated")
#
#    logger.info(f"\n📈 PREDICTIONS FOR THE NEXT {predict_days} DAYS:")
#
#    # Mostrar predicciones (ajusta según tu estructura)
#    if isinstance(predictions, dict):
#        linear_preds = predictions.get("linear", [])
#        ridge_preds = predictions.get("ridge", [])
#
#        if linear_preds:
#            logger.info("\n📊 Linear Regression:")
#            # Mostrar solo las predicciones futuras
#            future_start_idx = len(has_data) if len(linear_preds) > len(has_data) else 0
#            for i, p in enumerate(linear_preds[future_start_idx:], start=1):
#                logger.info(f"   Day {i}: ${p:.2f}")
#
#        if ridge_preds:
#            logger.info("\n🎯 Ridge Regression:")
#            future_start_idx = len(has_data) if len(ridge_preds) > len(has_data) else 0
#            for i, p in enumerate(ridge_preds[future_start_idx:], start=1):
#                logger.info(f"   Day {i}: ${p:.2f}")
#    else:
#        # Fallback si predictions no es dict
#        logger.info("\n📈 Predictions:")
#        for i, p in enumerate(predictions, start=1):
#            logger.info(f"   Day {i}: ${p:.2f}")
#
#    return predictions
#
#
#
### ======================================= MAIN FUNCTION =================================================
#
#def main():
#    logger.info("=" * 60)
#    logger.info("BITCOIN PRICE PREDICTOR")
#    logger.info("=" * 60)
#
#    # Initialize components
#    pipeline, db_manager = initialize_components()
#
#    while True:
#        logger.info("\n" + "=" * 60)
#        logger.info(" MAIN MENU")
#        logger.info("=" * 60)
#        logger.info("\nSelect an option:")
#        logger.info("   1️⃣  STAGE 1: Fetch data from CoinGecko and save to DB")
#        logger.info("   2️⃣  STAGE 2: Train model and predict (using DB data) and show plots")
#        #logger.info("   3️⃣  STAGE 3: Plot real prices, linear regression model and ridge regression model")
#        #logger.info("   4️⃣  EXIT")
#        logger.info("   3️⃣  EXIT")
#
#        choice = input("\n   Your choice (1-4): ").strip()
#
#        if choice == "1":
#              run_stage1_fetch(pipeline)
#        elif choice == "2":
#            run_stage2_train_predict(pipeline)
#        #elif choice == "3":
#        #    plot_all_models(pipeline)
#        elif choice == "3":
#            logger.info("\n👋 Thank you for using Bitcoin Price Predictor!")
#            break
#        else:
#              logger.info("❌ Invalid option. Please choose 1-4.")
#    
#    if db_manager:
#        db_manager.close()
#        logger.info(" Database connection closed")
#
#    logger.info("\n Program finished")
#
## ========================================= EXECUTION ===========================================
#
#if __name__ == "__main__":
#    main()