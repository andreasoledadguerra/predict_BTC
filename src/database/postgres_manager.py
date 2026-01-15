import pandas as pd
import logging

from sqlalchemy import create_engine,text
from typing import Optimal, Dict, Any
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
        self._engine = None
        self.logger = logging.getLogger(__name__)
    



