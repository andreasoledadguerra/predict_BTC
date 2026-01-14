from datetime import datetime, timezone

class DateConverter:

    @staticmethod
    def str_to_timestamp(date_str:str) -> int:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    @staticmethod
    # Convert date string to timestamp Unix
    def convert_to_unix(start_date:str, end_date: str) -> tuple[int, int]:
        start_ts = DateConverter.str_to_timestamp(start_date)
        end_ts =DateConverter.str_to_timestamp(end_date)
        return start_ts, end_ts