# Plant Disease / Pest Damage Detection — Exam 3 Own ML Project

Multi-class classification of leaf disease and pest-damage symptoms
(PlantVillage dataset), framed as the recognition module for a close-range
crop-scouting drone (orchard/vineyard row-scanning, which flies a few meters
from canopy — a real, existing practice, unlike photographing individual
insects from normal survey altitude). Pure end-to-end ML — comparing a
logistic regression baseline against a from-scratch 2D-CNN, with Grad-CAM
interpretability.

**Project history:** this is the third dataset explored for the "pest
detection for agricultural drones" topic. IP102 (macro insect photography)
turned out to be unrealistic for any drone altitude. Agriculture-Vision
fixed the realism problem but drifted away from "pests" into generic field
patterns and required a 20GB download. PlantVillage fixes both: disease/pest
lesions are leaf-scale (visible from realistic close-range drone imaging,
unlike sub-pixel insects), and the dataset is ~2GB.

## Project structure

```
.
├── data/                          # put PlantVillage class folders here (not committed)
├── figures/                       # plots saved here when the notebook runs
├── notebooks/
│   └── plant_disease_detection.ipynb  # main deliverable — run this top to bottom
├── src/
│   ├── preprocessing.py           # dataset discovery (ImageFolder-style) + top-N class selection
│   ├── dataset.py                 # PyTorch Dataset, transforms, stratified split, class weights
│   ├── models.py                  # LogisticRegressionBaseline, CNN2D (from scratch)
│   ├── train.py                   # training loop, Adam + CrossEntropyLoss, early stopping
│   ├── evaluate.py                # accuracy/macro-F1, confusion matrix, model comparison
│   ├── interpret.py               # Grad-CAM for the 2D-CNN, overlaid on the actual leaf image
│   └── utils.py                   # seeding, device selection
├── smoke_test.py                  # generates synthetic leaf-lesion images, runs full pipeline
├── build_notebook.py              # (re)generates the notebook programmatically
├── requirements.txt
└── ML_Fundamentals_Reference.md   # formulas/definitions backing every design choice below
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Getting the data

Download [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset)
(any Kaggle mirror works — search "PlantVillage dataset", ~2GB, use the
"color" images if the mirror separates color/grayscale/segmented versions).
Place the class folders directly under `data/`:

```
data/Tomato___Late_blight/*.jpg
data/Tomato___healthy/*.jpg
data/Potato___Early_blight/*.jpg
... (38 classes total across 14 crops)
```

`src/preprocessing.py::index_dataset()` auto-detects this layout — no manual
setup needed beyond placing the folders.

## Running

```bash
jupyter notebook notebooks/plant_disease_detection.ipynb
```

Run top to bottom. Section 2 lets you adjust `N_CLASSES` (default 15) and
`MIN_PER_CLASS` (default 200) — or restrict to a single crop's classes (e.g.
all `Tomato___*`) for an even more focused comparison, noted in the
notebook's markdown if you want that framing instead.

### Verifying the pipeline before real data arrives

`smoke_test.py` generates a small synthetic image dataset (leaf-green
backgrounds with colored lesion-like patches standing in for disease
symptoms) and runs the entire pipeline end-to-end. Already run once during
setup, confirmed working — including Grad-CAM correctly centering on the
synthetic lesion spots rather than healthy leaf area:

```bash
python3 smoke_test.py
```

Useful again any time you edit `src/` and want a fast correctness check.

## Design decisions (already justified — see ML_Fundamentals_Reference.md for formulas)

| Choice | Why |
|---|---|
| Categorical cross-entropy (`CrossEntropyLoss`) | Standard loss for multi-class classification with a softmax output (§1.2) |
| Adam optimizer | Mini-batch GD + momentum + adaptive LR — extension of §2.3, not in the course slides by name, flag this in the report |
| Dropout + BatchNorm in the CNN | Regularization against overfitting (§8) |
| Global average pooling before the classifier head | Manageable parameter count AND the architecture Grad-CAM needs |
| Top-N class selection (default 15) | PlantVillage's 38-class, 14-crop taxonomy is broader than needed for a tractable exam project — explicit, citable scoping decision |
| ±15° rotation only (not 360°) | Leaves have a natural up/down orientation, unlike the aerial imagery explored in this project's earlier Agriculture-Vision attempt |
| Stratified split, class-weighted loss | Same imbalance-handling principles used throughout this project's earlier iterations |
| From-scratch CNN (no pretrained backbone) | Keeps the architecture unambiguously "your own project" |

## Report checklist (5–8 pages)

- [ ] Problem relevance — why close-range drone crop scouting is a real practice, not hypothetical (see notebook §1)
- [ ] Data choice — source, size, quality, controlled-conditions limitation, "disease vs. pest" scope note (§1)
- [ ] Model setup justification — preprocessing/augmentation (§2) + architecture (§5)
- [ ] Baseline comparison — logistic regression vs. CNN (§4, §6)
- [ ] Evaluation — metrics + failure cases (§6, §7)
- [ ] Interpretation — Grad-CAM findings (§8)
- [ ] Conclusion — why this data/setup were right (or not) (§9 prompts in notebook, including an honest look back across all three dataset attempts)
