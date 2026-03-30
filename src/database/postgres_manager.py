import os
import pandas as pd
import logging

from sqlalchemy import create_engine,text
from typing import Dict, Any
from contextlib import contextmanager

from src.config.settings import PostgresSettings

class DatabaseManager:

    def __init__(
            self,
            settings: PostgresSettings
        ):
        
        self.settings = settings 
        self._engine = None # Lazy initialization
        self.logger = logging.getLogger(__name__)
    
    
    # Controlled access to the database engine
    @property
    def engine(self):

        if self._engine is None:
            connection_string = self.settings.get_connection_string()
            try:
                self._engine = create_engine(connection_string)

                with self._engine.connect() as conn:
                    conn.execute(text("SELECT 1"))
                self.logger.info(f"Conected to {self.settings.database} in {self.settings.host}: {self.settings.port}")
            except Exception as e:
                self.logger.error(f"Error connecting to PostgreSQL: {e}")
                raise
        return self._engine
    

    def save_btc_prices(self, df:pd.DataFrame, table_name: str = "btc_prices") -> int:
        try:
            required_columns = ['date', 'price_usd', 'volume_usd']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"DataFrame must contain columns: {required_columns}")
            
            #convert date compatible with SQL format
            df['date'] = pd.to_datetime(df['date']).dt.date

            records = list(df[required_columns].itertuples(index=False, name=None))


            insert_query = f"""
            INSERT INTO {table_name} (date, price_usd, volume_usd)
            VALUES (%s, %s, %s)
            ON CONFLICT (date) DO UPDATE SET 
                price_usd = EXCLUDED.price_usd, 
                volume_usd = EXCLUDED.volume_usd;
            """.format(table_name)

            # Execute the block insertion
            with self.engine.begin() as conn:
                conn.execute(insert_query, records)

            records_saved = len(df)
            self.logger.info(f"{records_saved} records upserted into {table_name}")
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
        SELECT date, price_usd, volume_usd
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
            #DEBUG
            self.logger.info(f"Returning DataFrame with columns: {df.columns.tolist()}")


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