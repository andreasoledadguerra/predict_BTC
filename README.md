
# Bitcoin Price Predictor (CLI)

This project is a Command Line Interface (CLI) application that retrieves historical Bitcoin price data, stores it in a database, trains a predictive model, and generates future price predictions based on user-defined parameters.

The application is designed as an end-to-end data pipeline, covering data acquisition, persistence, model training, and prediction, all operated through a simple and interactive terminal interface.


## Features

-Fetches historical Bitcoin price data from CoinGecko
-Stores retrieved data in a database for reuse and reproducibility
-Allows flexible date range selection for both data fetching and model training
-Trains a regression-based model using historical data
-Predicts future Bitcoin prices for a user-defined number of days
-Provides clear previews of fetched and training datasets directly in the CLI
-Fully interactive and easy to run from the terminal


## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/andreasoledadguerra/predict_BTC.git
   cd predict_BTC
   ```

2. It is recommended touse a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate   # Linux / macOS
    # venv\Scripts\activate    # Windows
    ```
 
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```


## Prerequisites

Before running the application make sure you have:
 - Docker installed and running
 - Docker Compose available
 - Pyhon 3.x installed


## Dependencies

The core Python dependencies include:

 - pandas -data manipulation and analysis
 - numpy -numerical computations
 - requests -HTTP requests to external APIs(CoinGecko)
 - psycopg2(or equivalent) - PostgreSQL database driver
 - scikit-learn -regression model training and prediction
 - SQLAlchemy -database abstraction and ORM

 The exact database driver depends on the database configured in Docker.

## Usage

1. Start the Docker container:
The database used by the application runs inside a Docker container and must be startes first:
  ```bash
  docker compose up -d
  ```
Make sure the container is running before proceeding.

2. Run the CLI Application:
   ```bash
   python3 predict_BTC/scripts/fetch_coingecko.py
   ```
Follow the on-screen instructions to:
 - Select date ranges
 - Train the model
 - Generate price predictions


## Project Structure

- `src/` : Core application logic
- `src/api`: A client for fetching historical Bitcoin price data from the CoinGecko API.
- `src/db/` : Database-related files and persisted data
- `src/ml/` : Data models and prediction logic
- `src/config`: Settings manager for database and API configurations using environment variables.
- `src/pipeline`: Data pipeline class
- `src/utils`: Utility for converting human-readable dates to Unix timestamps used by APIs.
- `docker/` : Docker-related configuration
- `docker/env/` : Docker environment variables
- `src/main.py` : CLI script to fetch data and run predictions
- `.env` : Environment variables (not committed)
- `docker-compose.yml` : Docker services definition (database)
- `.gitignore` : Git ignore rules
- `requirements.txt` : Python dependencies
- `README.md` : Project documentation
- `LICENSE` : Project license


## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the terms specified in the LICENSE file.
