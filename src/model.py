import xgboost as xgb
from sklearn.metrics import accuracy_score

def xgb_train(X_train, y_train, X_test, y_test, best_params,
              num_boost_round=800, early_stopping_rounds=50, eval_metric="auc"):
    # Konwersja kategorii
    for c in X_train.select_dtypes(include=["object"]).columns:
        X_train[c] = X_train[c].astype("category")
    for c in X_test.select_dtypes(include=["object"]).columns:
        X_test[c] = X_test[c].astype("category")
    
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)
    
    evals = [(dtrain, "train"), (dtest, "test")]
    
    print("Rozpoczynam trening XGBoost...")
    
    booster = xgb.train(
        {**best_params, "eval_metric": eval_metric, "tree_method": "hist"},
        dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=10
    )
    
    # Predykcje na teście dla finalnej accuracy
    y_pred_prob = booster.predict(dtest)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    print(f"Finalne Accuracy na zbiorze testowym: {acc:.4f}")
    
    return booster

