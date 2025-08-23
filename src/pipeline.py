import sys
from pathlib import Path

from src.data_loading import load_data, merge_data


sys.path.append(str(Path().resolve().parent)) # Katalog główny repo do sys.path

transactions_df, customers_df, articles_df = load_data()
df = merge_data(transactions_df, articles_df)