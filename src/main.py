import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv

from .config.settings import get_postgres_settings, get_coingecko_settings

from .api.coingecko_client import CoinGeckoClient
from .database.postgres_manager import DatabaseManager
from .ml.btc_predictor import BTCPredictor
from .pipeline.btc_pipeline import BTCDataPipeline


load_dotenv()

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
def run_stage1_fetch(pipeline):
    """
    Execute STAGE 1: Get data from CoinGecko and save to DB.
    """
    print("\n" + "=" * 60)
    print(" STAGE 1: FETCH DATA FROM COINGECKO")
    print("=" * 60)
    
    print("\n INSTRUCTIONS:")
    print("   - Maximum 90 days for CoinGecko free API")
    print("   - Data will be saved to PostgreSQL")
    print("   - Run this FIRST before making predictions")
    
    print("\n Enter date range to DOWNLOAD:")

    fetch_start = input(" Start date (YYYY-MM-DD): ").strip()
    fetch_end = input(" End date (YYYY-MM-DD): ").strip()

    result_fetch = pipeline.fetch_data(fetch_start, fetch_end)

    result_save =pipeline.save_data_in_db(result_fetch)


    print("\n" + "-" * 40)
    print(" STAGE 1 RESULTS")
    print("-" * 40)

    print(f" SUCCESS: {result_save} records saved")



def run_stage2_train_predict(pipeline):
    """
    Execute STAGE 2: Get data from DB, train model and predict.
    """

    print("\n" + "=" * 60)
    print(" STAGE 2: TRAIN MODEL AND PREDICT")
    print("=" * 60)

    print("\n Enter date range to TRAIN the model:")
    print("   (Must be within the available range)")

    train_start = input(" Start date (YYYY-MM-DD): ").strip()
    train_end = input(" End date (YYYY-MM-DD): ").strip()

    print(f"\n Checking data for {train_start} to {train_end}...")
    has_data = pipeline.get_data_for_training(train_start, train_end)

    days_input = input("\n How many days do you want to predict? (10): ").strip()
    predict_days = int(days_input) if days_input else 10

    print(f"\n Training model with data from {train_start} to {train_end}...")
    print(f" Predicting {predict_days} days into the future...")

    predictions = pipeline.predict_training_data(has_data, train_start, train_end, predict_days)


    print("\n" + "=" * 60)
    print(" STAGE 2 RESULTS")
    print("=" * 60)

    print(f" SUCCESS: Model trained and {predict_days} predictions generated")

    print(f"\n PREDICTIONS FOR THE NEXT {predict_days} DAYS:")
    
    for i, p in enumerate(predictions, start=1):
        print(f" Day {i}: {p:.2f}")

    return predictions

# ================================== MAIN FUNCTION =================================================

def main():
    print("=" * 60)
    print("BITCOIN PRICE PREDICTOR")
    print("=" * 60)

    # Initialize components
    pipeline, db_manager = initialize_components()

    while True:
        print("\n" + "=" * 60)
        print(" MAIN MENU")
        print("=" * 60)
        print("\nSelect an option:")
        print("   1️⃣  STAGE 1: Fetch data from CoinGecko and save to DB")
        print("   2️⃣  STAGE 2: Train model and predict (using DB data)")
        print("   3️⃣  Exit")

        choice = input("\n   Your choice (1-3): ").strip()

        if choice == "1":
              run_stage1_fetch(pipeline)
        elif choice == "2":
            run_stage2_train_predict(pipeline)
        elif choice == "3":
            print("\n👋 Thank you for using Bitcoin Price Predictor!")
            break
        else:
              print("❌ Invalid option. Please choose 1-4.")
    
    if db_manager:
        db_manager.close()
        print(" Database connection closed")

    print("\n Program finished")

# ========================================= EXECUTION ===========================================

if __name__ == "__main__":
    main()