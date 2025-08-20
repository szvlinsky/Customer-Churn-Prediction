from pathlib import Path
from typing import Dict, Any, List, Tuple
import tomllib
import pandas as pd
import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args


def load_xgb_bayes_config() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    repo_root = Path(__file__).resolve().parents[1]
    toml_file = repo_root / "pyproject.toml"

    with open(toml_file, "rb") as f:
        cfg_all = tomllib.load(f)
    xgb_bayes = cfg_all.get("tool", {}).get("xgb_bayes")

    required = [
        "n_trials",
        "random_state",
        "early_stopping_rounds",
        "processed_dir",
        "target_name",
        "objective_name",
    ]

    raw_space = xgb_bayes.get("space")

    space: List[Dict[str, Any]] = []
    for name, spec in raw_space.items():
        t = spec.get("type")
        if t == "cat":
            values = spec.get("values")
            space.append({"name": name, "type": "cat", "values": values})
        else:
            low = spec.get("lower"); high = spec.get("upper"); log = bool(spec.get("log", False))
            space.append({"name": name, "type": t, "low": low, "high": high, "log": log})

    cfg = {k: xgb_bayes[k] for k in required}

    return cfg, space


def load_split_data():
    project_root = Path(__file__).resolve().parents[1]
    split_path = project_root / "data" / "split"
    print("Split data path:", split_path.relative_to(project_root))

    X_train = pd.read_csv(split_path / "X_train.csv")
    y_train = pd.read_csv(split_path / "y_train.csv").squeeze("columns")
    X_val   = pd.read_csv(split_path / "X_val.csv")
    y_val   = pd.read_csv(split_path / "y_val.csv").squeeze("columns")
    X_test  = pd.read_csv(split_path / "X_test.csv")
    y_test  = pd.read_csv(split_path / "y_test.csv").squeeze("columns")

    for df in (X_train, X_val, X_test):
        for c in df.select_dtypes(include=["object"]).columns:
            df[c] = df[c].astype("category")

    return X_train, y_train, X_val, y_val, X_test, y_test


def tune_xgb_hyperparams(X_val, y_val, cfg, space_spec):
    Xv = X_val.copy()
    for c in Xv.select_dtypes(include=["object"]).columns:
        Xv[c] = Xv[c].astype("category")
    
    # Podział dla strojenia
    X_tr, X_ho, y_tr, y_ho = train_test_split(
        Xv, y_val, test_size=0.2,
        random_state=int(cfg["random_state"]),
        stratify=y_val
    )
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dhold = xgb.DMatrix(X_ho, label=y_ho, enable_categorical=True)
    
    # Przygotowanie przestrzeni hiperparametrów
    dims, names = [], []
    for s in space_spec:
        names.append(s["name"])
        if s["type"] == "int":
            dims.append(Integer(int(s["low"]), int(s["high"]), name=s["name"]))
        elif s["type"] == "float":
            dims.append(Real(float(s["low"]), float(s["high"]),
                             prior="log-uniform" if s.get("log", False) else "uniform",
                             name=s["name"]))
        else:
            dims.append(Categorical(list(s["values"]), name=s["name"]))
    
    base = {
        "objective": cfg.get("objective_name", "binary:logistic"),
        "eval_metric": "auc",
        "seed": int(cfg["random_state"]),
        "tree_method": "hist"
    }
    
    best_auc = -1.0
    trial_idx = 0

    # Funkcja celu
    @use_named_args(dims)
    def obj_fn(**hp):
        nonlocal best_auc, trial_idx
        trial_idx += 1
        t0 = time.time()
        booster = xgb.train({**base, **hp}, dtrain,
                            num_boost_round=int(cfg.get("num_boost_round", 800)),
                            evals=[(dhold, "hold")],
                            early_stopping_rounds=int(cfg.get("early_stopping_rounds", 50)),
                            verbose_eval=False)
        best_iter = getattr(booster, "best_iteration", None)
        preds = booster.predict(dhold, iteration_range=(0, (best_iter or 0)+1))
        auc = roc_auc_score(y_ho, preds)
        if auc > best_auc:
            best_auc = auc
        print(f"[trial {trial_idx}/{int(cfg.get('n_trials', 30))}] "
              f"auc={auc:.6f} | best={best_auc:.6f} | params={hp} | time={time.time()-t0:.2f}s")
        return -auc

    # Optymalizacja bayesowska
    res = gp_minimize(obj_fn, dims,
                      n_calls=int(cfg.get("n_trials", 30)),
                      random_state=int(cfg["random_state"]),
                      acq_func="EI")
    
    # Najlepsze parametry
    best_params = {n: v for n, v in zip(names, res.x)}
    return best_params
