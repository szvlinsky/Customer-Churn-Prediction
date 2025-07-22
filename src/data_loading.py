from pathlib import Path
import pandas as pd

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data" / "raw"
    print("Data path:", data_path.relative_to(project_root))

    transactions_df = pd.read_csv(data_path / "transactions_train.csv")
    customers_df = pd.read_csv(data_path / "customers.csv")
    articles_df = pd.read_csv(data_path / "articles.csv")

    return transactions_df, customers_df, articles_df

def merge_data(transactions_df: pd.DataFrame, articles_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = (
        transactions_df
        .merge(articles_df, on="article_id", how="left")
    )
    return merged_df