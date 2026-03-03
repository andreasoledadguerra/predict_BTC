# script para cargar precio histórico del BTC

import pandas as pd
import sqlite3 

# Read .csv file
df= pd.read_csv('BTCUSD_1d_Binance.csv')

# Convert column to datetime
df['Open time']=pd.to_datetime(df['Open time'])

#