# PM2.5 One-Hour-Ahead Forecasting

Predicts PM2.5 concentration one hour ahead for 12 Beijing monitoring stations.

## Files

- `train.csv`, `test.csv` — raw data
- `feature_engineering.py` — lag/rolling/spatial/stability feature engineering
- `train_lightgbm.py` — LightGBM model (huber loss, 13,000 rounds)
- `train_gru.py` — GRU sequence model
- `run_all.py` — **entry point**, runs the full pipeline end to end and produces `submission.csv`
- `model_report.pdf` — design rationale (feature engineering, model choice, hyperparameters)

## Usage

```bash
pip install lightgbm torch pandas numpy scikit-learn
python run_all.py
```

Produces `submission.csv` in this folder.
