import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class PostgresSettings:
    user: str
    password: str
    database: str
    host: str = "localhost"
    port: int = 5433

    def get_connection_string(self):
        connection_string = (
            f"postgresql+psycopg2://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )
        return connection_string

@dataclass(frozen=True)
class CoinGeckoSettings:
    API_KEY:str

def get_postgres_settings() -> PostgresSettings:
    return PostgresSettings(
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
        database=os.getenv("POSTGRES_DB"),
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", 5433)),

    )

def get_coingecko_settings() -> CoinGeckoSettings:
    return CoinGeckoSettings(
        API_KEY=os.getenv("COINGECKO_API_KEY")
    )
