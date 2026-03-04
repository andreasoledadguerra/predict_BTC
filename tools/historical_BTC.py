# script para cargar precio histórico del BTC
import os
import logging
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
from src.config.settings import get_postgres_settings
from src.database.postgres_manager import DatabaseManager
from src.config.logs import configure_logging

logger = logging.getLogger(__name__)

load_dotenv()
configure_logging()

#load settings
pg_settings = get_postgres_settings()

#Initialize
db_manager = DatabaseManager(settings=pg_settings)

#Activate the connection
engine =db_manager.engine

# Read .csv file
df= pd.read_csv('BTCUSD_1d_Binance.csv')

# Convert column to datetime
df['Open time']=pd.to_datetime(df['Open time'])

# Define range date
start_date = '2025-01-01'
end_date = '2025-09-30'

# Create mask using the column date
mask = (df['Open time'] >= start_date) & (df['Open time'] <= end_date)
df_filtered = df.loc[mask].copy()


# Create DataFrame with needed columns

df_canonico = pd.DataFrame({
    'date': df_filtered['Open time'].dt.date,
    'price_usd': df_filtered['Close'],
    'asset': 'BTC'
})

logger.info(f"Filas a insertar: {len(df_canonico)}")
logger.info("Primeras 3 filas:")
logger.info(df_canonico.head(3))

df_canonico.to_sql(
    'btc_prices',
    con=engine,
    if_exists='append',
    index=False
)

logger.info("Datos insertados correctamente en la tabla btc_prices.")