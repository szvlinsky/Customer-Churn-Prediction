from __future__ import annotations
from typing import Tuple
import pandas as pd
import tomllib
from pathlib import Path

def load_data_and_config() -> Tuple[pd.DataFrame, dict]:
    repo_root = Path(__file__).resolve().parents[1]
    toml_file = repo_root / "pyproject.toml"
    if not toml_file.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {toml_file}")

    with open(toml_file, "rb") as f:
        cfg_all = tomllib.load(f)
    cfg = cfg_all.get("tool", {}).get("split", {})

    if (cfg.get("valid_size", 0) + cfg.get("test_size", 0)) >= 100:
        raise ValueError("valid_size + test_size musi być < 100")

    in_path = repo_root / cfg.get("input_path", "data/processed/final_df.csv")
    if not in_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku danych: {in_path}")

    in_path.suffix.lower() == ".csv"
    df = pd.read_csv(in_path)
    
    return df, cfg
