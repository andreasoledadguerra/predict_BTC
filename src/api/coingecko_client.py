import requests
from typing import Dict, Optional
from datetime import datetime, timezone
import time

class CoinGeckoClient:

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, API_KEY, timeout):

        self.api_key = API_KEY
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"x-cg-demo-api-key": API_KEY})


        def _build_request_config(start_ts: int, end_ts: int) -> dict:
            
            config = {
                "url":  f"{self.BASE_URL}/coins/bitcoin/market_chart/range",
                "params": {      
                    "vs_currency": "usd",
                    "from": start_ts,
                    "to": end_ts
                }
            }
        
            return config


       # url = f"{self.BASE_URL}/coins/bitcoin/market_chart/range"

       # params = {
       #     "vs_currency": currency,
       #     "from": start_ts,
       #     "to": end_ts
       # } 

        time.sleep(1) #basic rate limiting

        response = self.session.get(
            url,
            params=params,
            timeout=self.timeout
        )
        response.raise_for_status()

        return response.json()
    

    def str_to_timestamp(self, date_str:str) -> int:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())
    
    
    def close(self):
        self.session.close()
        
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()