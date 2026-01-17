import numpy as np
import pandas as pd
import logging
from sklearn.linear_model import LinearRegression
from typing import Tuple, Optional, List

class BTCPredictor:

    def __init__(self, model: Optional[LinearRegression] = None):

        self.model = model or LinearRegression()
        self.logger = logging.getLogger(__name__)
        self.is_trained = False
        self.training_size = 0 