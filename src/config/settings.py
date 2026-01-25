import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class PostgresSettings:
    user: str
    password: str
    database: str
    port : int

@dataclass(frozen=True)
class CoinGeckoSettings:
    API_KEY:str