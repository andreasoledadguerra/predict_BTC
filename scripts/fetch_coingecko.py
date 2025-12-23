import os
import requests

from datetime import datetime, timezone
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("COINGECKO_API_KEY")
print(f"API Key loaded: {'Yes' if API_KEY else 'No'}")
print(f"API Key length: {len(API_KEY) if API_KEY else 0} characters")
if API_KEY:
    print(f"Key starts with: {API_KEY[:8]}...")

# Function to get UNIX date from CoinGecko 
def to_timestamp(date_str:str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    timestamp_seconds = int(dt.timestamp())
    timestamp_ms = timestamp_seconds * 1000
    print(f"{date_str} -> {timestamp_seconds}s -> {timestamp_ms}ms")

start_ts = to_timestamp("2024-10-01")
end_ts = to_timestamp("2024-10-07")
#esta función sirve para recibir fecha inicial y fecha final

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