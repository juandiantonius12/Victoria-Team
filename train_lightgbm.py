"""
Train the LightGBM component of the model.

The target is the raw PM2_5_next_hour level. Huber loss is used
instead of squared error, since PM2.5 has occasional large, sharp
spikes that would otherwise dominate a squared-error objective.

Hyperparameters were selected via time-based validation.
"""
import os
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

from feature_engineering import build_full_feature_set

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "PM2_5_next_hour"
N_ROUNDS = 13000

LGB_PARAMS = {
    "objective": "huber",
    "alpha": 40.0,
    "metric": "rmse",
    "learning_rate": 0.03,
    "num_leaves": 127,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.1,
    "lambda_l2": 0.5,
    "max_depth": -1,
    "seed": 42,
    "num_threads": os.cpu_count(),
    "verbose": -1,
}


def main():
    t0 = time.time()

    train = pd.read_csv(os.path.join(HERE, "train.csv"), parse_dates=["observation_timestamp"])
    test = pd.read_csv(os.path.join(HERE, "test.csv"), parse_dates=["observation_timestamp"])

    F = build_full_feature_set(train, test, TARGET)
    feature_cols = [c for c in F.columns if c not in {"observation_timestamp", "_split", "id", TARGET}]

    TR = F[F._split == "train"].copy()
    TE = F[F._split == "test"].copy()
    print(f"Feature set ready: {len(feature_cols)} features, {len(TR)} train rows, {len(TE)} test rows  [{time.time()-t0:.1f}s]")

    dtrain = lgb.Dataset(TR[feature_cols], TR[TARGET], categorical_feature=["station"])
    model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=N_ROUNDS)
    pred = np.clip(model.predict(TE[feature_cols]), 0, None)
    print(f"Model trained  [{time.time()-t0:.1f}s]")

    out_path = os.path.join(HERE, "lightgbm_predictions.csv")
    pd.DataFrame({"id": TE.id.values, TARGET: pred}).to_csv(out_path, index=False)
    print(f"Saved {out_path}")
    print(f"Elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
