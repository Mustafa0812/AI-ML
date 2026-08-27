# Engine Condition Prediction from Sensor Data — Exam 3 Own ML Project

Binary classification of engine condition (healthy/faulty) from six live engine
sensor readings (RPM, lubricating oil pressure, fuel pressure, coolant
pressure, lubricating oil temperature, coolant temperature). A from-scratch
MLP is compared against logistic regression baselines, with a Gaussian-noise
augmentation ablation and permutation-importance interpretation.

## Project structure

```
.
├── data/
│   └── engine_data.csv            # committed — no download needed
├── notebooks/
│   └── main_analysis.ipynb        # main deliverable — run this top to bottom
├── src/
│   ├── preprocess.py              # load/split/scale, class-weight, augmentation
│   ├── baselines.py               # logistic regression baselines
│   ├── model.py                   # EngineConditionMLP architecture
│   ├── train.py                   # training loop, early stopping
│   └── evaluate.py                # metrics, plots, permutation importance
├── outputs/
│   ├── figures/                   # confusion matrices, ROC curves, importance plot
│   └── metrics.json               # metrics used by the report
├── report/
│   └── exam3_report.pdf           # the written report
├── exam_3_own_project.pdf         # assignment brief
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Running

The dataset is already committed under `data/`, so no download step is
needed.

```bash
jupyter notebook notebooks/main_analysis.ipynb
```

Run top to bottom. It loads the data, builds the 70/15/15 stratified split,
trains the logistic regression baselines and the MLP (with and without
augmentation), and reproduces every table/figure used in the report,
saving figures to `outputs/figures/` and printing the comparison metrics.

## Report

The full write-up (problem relevance, data audit, model/architecture
justification, baseline comparison, evaluation, interpretation, and
conclusion) is in [`report/exam3_report.pdf`](report/exam3_report.pdf).
