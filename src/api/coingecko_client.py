import requests
import pandas as pd
import time
import logging

from typing import Dict, Optional
from datetime import datetime, timezone
from src.utils.date_converter import DateConverter



class CoinGeckoClient:
    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, API_KEY: str, timeout: int):

        self.api_key = API_KEY
        self.logger = logging.getLogger(__name__)
        self.timeout = timeout
        self.session = requests.Session()

        self.session.headers.update({"x-cg-demo-api-key": API_KEY})


        def build_request_config(self,start_ts: int, end_ts: int) -> dict:
            
            config = {
                "url":  f"{self.BASE_URL}/coins/bitcoin/market_chart/range",
                "params": {      
                    "vs_currency": "usd",
                    "from": start_ts,
                    "to": end_ts
                }
            }
        
            return config


        def parse_price_data(self, data:dict) -> pd.DataFrame:

            if "prices" not in data:
                raise ValueError ("La respuesta no contiene 'prices'")

            df = pd.DataFrame(data["prices"], columns=["timestamp_ms", "price_usd"])

            # Transformar timestamp a fecha (borrar)
            df["date"] = pd.to_datetime(df["timestamp_ms"], unit="ms").dt.date

            # Agregar columna de activo
            df["asset"] = "BTC"

            # Reordenar columnas
            df = df[["date", "price_usd", "asset"]]

            return df
    

        def str_to_timestamp(self, date_str:str) -> int:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp())

        
        def fetch_bitcoin_prices(self, start_date: str, end_date: str) -> pd.DataFrame:

            self.logger.info(f"Getting data from {start_date} to {end_date}...")

            start_ts, end_ts = DateConverter.convert_to_unix(start_date, end_date)

            request_config = self._build_request_config(start_ts, end_ts)

            time.sleep(1)
    
            response = self.session.get(
                request_config["url"],
                params=request_config["params"],
                timeout=self.timeout
                
            )
            response.raise_for_status()

            raw_data = response.json()

            df = self._parse_price_data(raw_data)

            self.logger.info(f" {len(df)} obtained records")
            return df
        

        def close(self):
            self.session.close()
        
        def __exit__(self, exc_tye, exc_val, exc_tb):
            self.close()