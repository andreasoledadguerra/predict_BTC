import os
import requests

from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("COINGECKO_API_KEY")

end_ts = int(datetime.now(timezone.utc).timestamp())
start_ts = int((datetime.now(timezone.utc) - timedelta(days=7)).timestamp())

#Call the API
url =  "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"

params = {
    "vs_currency": "usd",
    "from": start_ts,
    "to": end_ts
}


headers = {
    "accept": "application/json",
    "x-cg-demo-api-key": API_KEY
}
response = requests.get(url, params=params)
#esponse.raise_for_status()
data = response.json()


print("Keys:", data.keys())
print("First price entry:", data["prices"][0])