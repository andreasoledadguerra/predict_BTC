import os
import pandas as pd
import logging

from sqlalchemy import create_engine,text
from typing import Dict, Any
from contextlib import contextmanager

class DatabaseManager:

    def __init__(
            self,
            user: str,
            password: str,
            database: str,
            host: str = "localhost",
            port: str = "5433"
    ):
        
        self.user = user
        self.password = password 
        self.database = database 
        self.host = host
        self.port = port 
        self._engine = None # Lazy initialization
        self.logger = logging.getLogger(__name__)
    
    
    # Controlled access to the database engine
    @property
    def engine(self):

        if self._engine is None:
            connection_string = (
                f"postgresql+psycopg2://{self.user}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
            )
            try:
                self._engine = create_engine(connection_string)

                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                self.logger.info(f"Conected to {self.database} in {self.host}: {self.port}")
            except Exception as e:
                self.logger.error(f"Error connecting to PostgreSQL: {e}")
                raise
        return self._engine
    

    def save_btc_prices(self, df:pd.DataFrame, table_name: str = "btc_prices") -> int:

        try:
            df.to_sql(
                table_name,
                self.engine,
                if_exists="append",
                index=False,
                method='multi',
                chunksize=1000
            )

            records_saved = len(df)
            self.logger.info(f"{records_saved} saved records in {table_name}")
            return records_saved
        
        except Exception as e:
            self.logger.error(f"Error saving in database: {e}")
            raise

    def get_btc_prices(
        self,
        start_date: str,
        end_date: str,
        table_name: str = "btc_prices"
    ) -> pd.DataFrame:
        
        query = text(f"""
        SELECT date, price_usd 
        FROM {table_name} 
        WHERE date BETWEEN :start_date AND :end_date 
        ORDER BY date
        """)

        try:
            df = pd.read_sql(
                query,
                self.engine,
                params={"start_date": start_date, "end_date": end_date}
            )
            self.logger.info(f"{len(df)} data obtained from {table_name}")
            return df
        
        except Exception as e:
            self.logger.error(f"Database read permission error: {e}")
            raise
        

    @contextmanager
    def get_connection(self):

        conn = self.engine.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


    def close(self):

        if self._engine:
            self._engine.dispose()
            self.logger.info("Connection to database closed")