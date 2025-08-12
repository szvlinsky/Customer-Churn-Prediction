from __future__ import annotations
from typing import Tuple
import pandas as pd
import numpy as np
import tomllib
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_data_and_config() -> Tuple[pd.DataFrame, dict]:
    repo_root = Path(__file__).resolve().parents[1]
    toml_file = repo_root / "pyproject.toml"

    with open(toml_file, "rb") as f:
        cfg_all = tomllib.load(f)
    cfg = cfg_all.get("tool", {}).get("split", {})

    if (cfg.get("valid_size", 0) + cfg.get("test_size", 0)) >= 100:
        raise ValueError("valid_size + test_size musi być < 100")

    in_path = repo_root / cfg.get("input_path", "data/processed/final_df.csv")
    in_path.suffix.lower() == ".csv"
    df = pd.read_csv(in_path)
    
    return df, cfg


def split_data(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                     pd.Series, pd.Series, pd.Series]:
    test_pct   = float(cfg["test_size"])  / 100.0
    valid_pct  = float(cfg["valid_size"]) / 100.0
    rs         = int(cfg.get("random_state", 42))

    y = df["churn"]
    X = df.drop(columns=["churn", "customer_id"])

    n = len(df)
    n_test  = int(round(n * test_pct))
    n_valid = int(round(n * valid_pct))

    rng = np.random.RandomState(rs)

    idx_test_parts  = []
    idx_valid_parts = []
    idx_train_parts = []

    for c in y.unique():
        idx_c = np.where(y.values == c)[0]
        rng.shuffle(idx_c)

        n_c = len(idx_c)
        n_test_c  = int(round(n_c * test_pct))
        n_valid_c = int(round(n_c * valid_pct))

        idx_test_parts.append(idx_c[:n_test_c])
        idx_valid_parts.append(idx_c[n_test_c:n_test_c + n_valid_c])
        idx_train_parts.append(idx_c[n_test_c + n_valid_c:])

    idx_test  = np.concatenate(idx_test_parts)
    idx_valid = np.concatenate(idx_valid_parts)
    idx_train = np.concatenate(idx_train_parts)

    X_train = X.iloc[idx_train].reset_index(drop=True)
    X_val   = X.iloc[idx_valid].reset_index(drop=True)
    X_test  = X.iloc[idx_test].reset_index(drop=True)
    y_train = y.iloc[idx_train].reset_index(drop=True)
    y_val   = y.iloc[idx_valid].reset_index(drop=True)
    y_test  = y.iloc[idx_test].reset_index(drop=True)

    return X_train, X_val, X_test, y_train, y_val, y_test
