import requests
from typing import Dict, Optional
import time

class CoinGeckoClient:

    BASE_URL = "https://api.coingecko.com/api/v3"

    def __init__(self, API_KEY, timeout):

        self.api_key = API_KEY
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"x-cg-demo-api-key": API_KEY})

