"""
Entry point: run the full pipeline end to end and produce submission.csv.

The final forecast blends two models with different inductive biases:
a LightGBM model over hand-built lag/rolling/spatial features (built
around PM10 and the other pollutants, including cross-station "donor"
features and atmospheric-stability proxies), and a GRU sequence model
reading raw per-station history directly. Averaging two models that
make different kinds of mistakes reduces the overall error more than
either model alone -- what matters for a blend is that the two models'
errors are only partially related, not that both are equally accurate
on their own.

The two stages are launched as separate processes (see train_gru.py
for why), so this script just runs each in turn and combines their
saved predictions.
"""
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = "PM2_5_next_hour"

# Blend weight selected via the same time-based validation described in
# train_lightgbm.py.
GRU_WEIGHT = 0.35


def run_stage(script_name):
    path = os.path.join(HERE, script_name)
    print(f"--- {script_name} ---", flush=True)
    subprocess.run([sys.executable, "-u", path], check=True, cwd=HERE)


def main():
    t0 = time.time()

    run_stage("train_lightgbm.py")
    run_stage("train_gru.py")

    lgb_pred = pd.read_csv(os.path.join(HERE, "lightgbm_predictions.csv")).rename(columns={TARGET: "pred_lgb"})
    gru_pred = pd.read_csv(os.path.join(HERE, "gru_predictions.csv")).rename(columns={TARGET: "pred_gru"})
    merged = lgb_pred.merge(gru_pred, on="id")

    final_pred = np.clip(GRU_WEIGHT * merged["pred_gru"] + (1 - GRU_WEIGHT) * merged["pred_lgb"], 0, None)

    out_path = os.path.join(HERE, "submission.csv")
    pd.DataFrame({"id": merged.id.values, TARGET: final_pred}).to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")
    print(f"Total elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
