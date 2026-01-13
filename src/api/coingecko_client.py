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

    def get_bitcoin_range(
        self,
        start_date: str,
        end_date: str,
        currency: str = "usd"
    ) -> Dict:
        
        start_ts = self.str_to_timestamp(start_date)
        end_ts = self.str_to_timestamp(end_date)

        url = f"{self.BASE_URL}/coins/bitcoin/market_chart/range"

        params = {
            "vs_currency": currency,
            "from": start_ts,
            "to": end_ts
        } 

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
    
    