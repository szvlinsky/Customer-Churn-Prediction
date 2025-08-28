import sys
from pathlib import Path
sys.path.append(str(Path().resolve().parent)) # Katalog główny repo do sys.path

from bayes_search import load_split_data, load_xgb_bayes_config, tune_xgb_hyperparams
from model import xgb_train

# 1. Ładowanie danych i ustawień
X_train, y_train, X_val, y_val, X_test, y_test = load_split_data()
cfg, space = load_xgb_bayes_config()
print("Konfiguracja:", cfg)
print("Przestrzeń hiperparametrów:")
for param in space:
    print(param)

# 2. Wyszukanie optymalnych hiperparametrów
best_params = tune_xgb_hyperparams(X_val, y_val, cfg, space)
print("Najlepsze hiperparametry:", best_params)

# 3. Trenowanie i test finalnego modelu
booster = xgb_train(X_train, y_train, X_test, y_test, best_params, 
                    num_boost_round=500, early_stopping_rounds=100)