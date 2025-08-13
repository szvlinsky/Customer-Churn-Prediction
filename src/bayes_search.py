from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import tomllib


from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Tuple
import tomllib


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
        low = spec.get("lower")
        high = spec.get("upper")
        log = bool(spec.get("log", False))
        space.append({"name": name, "type": t, "low": low, "high": high, "log": log})

    cfg = {k: xgb_bayes[k] for k in required}
    
    return cfg, space

