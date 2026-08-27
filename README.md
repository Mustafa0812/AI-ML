# Turbofan Engine RUL Prediction

Regression: predict Remaining Useful Life (RUL, in cycles) for NASA C-MAPSS
turbofan engines (subset FD001 — single operating condition, single fault
mode) from a windowed sequence of 15 sensor readings. An LSTM (which sees a
30-cycle window of sensor history) is compared against linear regression
baselines (which see only the current cycle), with a window-length ablation
and permutation-importance interpretation.

## Project structure

```
.
├── data/
│   └── CMaps/
│       ├── train_FD001.txt        # committed — no download needed
│       ├── test_FD001.txt
│       └── RUL_FD001.txt          # ground-truth test RUL, one value per unit
├── notebooks/
│   └── main_analysis.ipynb        # main deliverable — run this top to bottom
├── src/
│   ├── preprocess.py              # load, RUL capping, unit-level split, scaling, windowing
│   ├── baselines.py               # linear regression baselines (snapshot / cycle-only)
│   ├── model.py                   # TurbofanRULLSTM architecture
│   ├── train.py                   # training loop, early stopping
│   └── evaluate.py                # RMSE/MAE/PHM08 score, plots, permutation importance
├── outputs/
│   ├── figures/                   # pred-vs-actual, training curves, RUL-band error, trajectories
│   └── metrics.json               # metrics used by the report
├── report/
│   └── report.pdf                 # theoretical answers + full project write-up
├── exam_3_own_project.pdf         # assignment brief
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

No new dependencies were needed for the LSTM — PyTorch was already pinned in
`requirements.txt`.

## Running

The dataset is already committed under `data/CMaps/`, so no download step is
needed.

```bash
jupyter notebook notebooks/main_analysis.ipynb
```

Run top to bottom. It audits the raw data, builds the unit-level 80/20
train/val split and 30-cycle sliding windows, trains the linear baselines
and the LSTM (plus a window=15 ablation), and reproduces every table/figure
used in the report, saving figures to `outputs/figures/` and printing the
comparison metrics.

Equivalently, from `src/`: `python evaluate.py` regenerates everything
(`outputs/metrics.json` and all figures) without opening the notebook.

## Report

[`report/report.pdf`](report/report.pdf) contains the theoretical question
answers (model-family comparison; gradient descent) followed by the full
project write-up: problem relevance, data audit, model/architecture
justification, baseline comparison, evaluation, interpretation, limitations,
and conclusion.
