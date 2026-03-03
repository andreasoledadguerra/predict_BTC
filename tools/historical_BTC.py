# script para cargar precio histórico del BTC

import pandas as pd
from sqlalchemy import create_engine
import os

# Read .csv file
df= pd.read_csv('BTCUSD_1d_Binance.csv')

# Convert column to datetime
df['Open time']=pd.to_datetime(df['Open time'])

# Define range date
start_date = '2025-01-01'
end_date = '2025-09-30'

# Create mask using the column date
mask = (df['Open time'] >= start_date) & (df['Open time'] <= end_date)
df_filtered = df.loc[mask]

#print(f"Registros del rango: {len(df_filtered)}")
