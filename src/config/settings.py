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

# seguir con los métodos de PostgresSettings y CoinGeckoSettings

def get_postgres_settings() -> PostgresSettings:
    return PostgresSettings(
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB"),
        port=int(os.getenv("POSTGRES_PORT", 5433)),
    )

