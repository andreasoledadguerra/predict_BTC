import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv


from api.coingecko_client import CoinGeckoClient
from database.postgres_manager import DatabaseManager
from ml.btc_predictor import BTCPredictor
from pipeline.btc_pipeline import BTCDataPipeline

# Load variables defined in the .env file into the environment (delete)
load_dotenv()

# Retrieve the Postgres environment variables (delete)
POSTGRES_USER= os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD= os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB= os.getenv("POSTGRES_DB")
POSTGRES_PORT= os.getenv("POSTGRES_PORT")

# Retrieve the CoinGecko API key from environment variables(delete)
API_KEY = os.getenv("COINGECKO_API_KEY")


# ====================================== SETTING =================================================
def initialize_components(env_vars):

    api_client = CoinGeckoClient(api_key=env_vars["COINGECKO_API_KEY"])

    db_manager = DatabaseManager(
        user=env_vars["POSTGRES_USER"],
        password=env_vars["POSTGRES_PASSWORD"],
        database=env_vars["POSTGRES_DB"],
        port=env_vars["POSTGRES_PORT"]
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

    result = pipeline.execute_stage1_fetch_and_save(fetch_start, fetch_end)

    print("\n" + "-" * 40)
    print(" STAGE 1 RESULTS")
    print("-" * 40)

    print(f" SUCCESS: {result['records_saved']} records saved")



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
    has_data = pipeline.validate_date_range_in_db(train_start, train_end)

    days_input = input("\n How many days do you want to predict? (10): ").strip()
    predict_days = int(days_input) if days_input else 10

    print(f"\n Training model with data from {train_start} to {train_end}...")
    print(f" Predicting {predict_days} days into the future...")

    result = pipeline.execute_stage2_train_and_predict(train_start, train_end, predict_days)

    print("\n" + "=" * 60)
    print(" STAGE 2 RESULTS")
    print("=" * 60)

    print(f" SUCCESS: Model trained and {predict_days} predictions generated")

    print(f"\n PREDICTIONS FOR THE NEXT {predict_days} DAYS:")
    
    predictions = result.get("predictions", [])
    

    return predictions

# ================================= 