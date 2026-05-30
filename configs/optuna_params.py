SEARCH_SPACES = {
    "xgboost": {
        "n_estimators": (100, 500),
        "max_depth": (3, 10),
        "learning_rate": (1e-3, 0.3),
        "subsample": (0.6, 1.0),
        "colsample_bytree": (0.6, 1.0),
    },
    "lightgbm": {
        "n_estimators": (100, 500),
        "num_leaves": (8, 128),
        "learning_rate": (1e-3, 0.3),
        "subsample": (0.6, 1.0),
        "reg_alpha": (1e-3, 10.0),
    },
    "random_forest": {
        "n_estimators": (50, 300),
        "max_depth": (3, 15),
        "min_samples_split": (2, 20),
    },
}
