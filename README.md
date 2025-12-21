The PostgreSQL database is containerized using Docker Compose, with persistent volumes and environment-based configuration.

Se obtienen el rango de fechas del precio de bitcoin de CoinGecko (que están en formato UNIX), se convierten a timestamp y se genera una lista  (data) con los precios de dicho rango temporal.

data se pasa a un dataframe, y se guarda en Postgres.