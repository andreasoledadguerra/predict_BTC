import requests
from datetime import datetime, timezone

def to_timestamp(date_str:str) -> int:
    return int(
        datetime.strptime(date_str, "%Y-%m-%d")
        .timestamp()
    )