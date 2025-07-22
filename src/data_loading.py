import pandas as pd
from pathlib import Path

def load_data(data_dir: str = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data_path = Path(data_dir)

    transactions_df = pd.read_csv(data_path / "transactions_train.csv")
    customers_df = pd.read_csv(data_path / "customers.csv")
    articles_df = pd.read_csv(data_path / "articles.csv")

    return transactions_df, customers_df, articles_df

def merge_data(transactions_df: pd.DataFrame, customers_df: pd.DataFrame, articles_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = (
        transactions_df
        .merge(customers_df, on="customer_id", how="left")
        .merge(articles_df, on="article_id", how="left")
    )
    return merged_df