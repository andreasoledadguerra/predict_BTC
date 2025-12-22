import os
import requests

from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

# Function to get UNIX date from CoinGecko 
def to_timestamp(date_str:str) -> int:
    return int(
        datetime.strptime(date_str, "%Y-%m-%d")
        .replace(tzinfo=timezone.utc)
        .timestamp()
    )

start_ts = to_timestamp("2024-10-01")
end_ts = to_timestamp("2024-11-01")
#esta función sirve para recibir fecha inicial y fecha final

#Call the API
url =  "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart/range"

params = {
    "vs_currency": "usd",
    "from": start_ts,
    "to": end_ts
}

headers = {
    "x-cg-pro-api-key":os.getenv("COINGECKO_API_KEY")
}
response = requests.get(url, params=params, headers=headers)
response.raise_for_status()
data = response.json()

print("Keys:", data.keys())
print("First price entry:", data["prices"][0])