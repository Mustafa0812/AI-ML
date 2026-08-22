# Exam 3 — Own ML Project: Plant Disease / Pest Damage Detection (PlantVillage)

## Context
University "Exam 3" deliverable. Two theoretical questions (independent of
the project, already covered — see `ML_Fundamentals_Reference.md`) plus a
hands-on ML project.

**Deliverables required:** 5–8 page report, reproducible code/notebook,
short conclusion: "Why this data and setup were the right choice (or not)".

## Project Decision (topic history — 5th topic, 3rd dataset for "pest detection")

1. CWRU bearing fault classification — scoped, never built.
2. Exoplanet detection (Kepler light curves) — fully built, working, separate
   folder (`exoplanet_project/`).
3. Pest detection, **IP102** dataset — fully built, working, separate folder
   (`pest_project/`). Abandoned: user correctly identified IP102 images are
   macro/studio photography, unrealistic for any real drone altitude (an
   individual insect is sub-pixel from 10-50m up).
4. Pest detection, **Agriculture-Vision** dataset — fully built, working,
   separate folder (`agvision_project/`). Fixed the realism problem (genuine
   aircraft-captured aerial imagery) but user correctly flagged two new
   problems: (a) 20GB download, unreasonable for a course exam, (b) the
   dataset is about generic field anomalies (weeds, drydown, standing water,
   nutrient deficiency) — NOT pests, a real topic drift the user caught.
5. **Pest/disease detection, PlantVillage dataset — current, actively
   built.** Resolves both prior complaints: ~2GB (not 20GB), and it's
   genuinely about crop health/disease/pest-damage symptoms (not generic
   field patterns). Trade-off: images are close-range leaf photos under
   controlled conditions, not aerial. Framing used: this is the recognition
   module for CLOSE-RANGE drone crop scouting (orchard/vineyard row-scanning
   at a few meters from canopy), which is a real, existing precision-ag
   practice — genuinely different from, and more realistic than, the
   individual-insect-at-survey-altitude framing that sank IP102.

All five explorations share the same underlying technical shape (raw input
→ CNN, cross-entropy-family loss, baseline comparison, saliency-based
interpretation), so `ML_Fundamentals_Reference.md` applies unchanged.

**Constraint carried over: "pure ML model"** = no hand-crafted domain
features.

## Honest framing points already built into the notebook (don't lose these)

- PlantVillage covers disease/pest damage broadly (fungal, bacterial, viral,
  AND insect-caused symptoms) — stated explicitly in Section 1 that this is
  "crop health symptom detection," of which insect pest damage is one cause
  among several, not a pure "insect pest" dataset.
- Controlled-condition images (uniform background, consistent framing) are
  still a real domain gap vs. actual drone footage (variable lighting,
  motion blur, cluttered backgrounds) — smaller gap than IP102's (leaf-scale
  symptoms are visible at realistic close range, unlike individual insects),
  but stated plainly rather than glossed over.
- Section 9's conclusion prompts explicitly ask the user to reflect honestly
  across all three dataset attempts (IP102 → Agriculture-Vision →
  PlantVillage) — useful, genuine material for the report's conclusion.

## Technical Plan — IMPLEMENTED

**Task:** multi-class classification (default 15 classes from PlantVillage's
38, configurable) — healthy vs. which disease/damage pattern.

**Dataset:** PlantVillage (github.com/spMohanty/PlantVillage-Dataset) —
~54,300 images, 38 classes (14 crop species x disease/healthy combos),
ImageFolder-style distribution (`data/<ClassName>/*.jpg`) on every Kaggle
mirror — same format as IP102's Format A, which is why most code from
`pest_project/` ported over with minimal changes. **Status: not yet in
`data/`** in this environment — user downloads manually (Kaggle).

**Pipeline (all implemented in `src/`, mostly reused verbatim or
lightly-renamed from `pest_project/` since the classification-once-you-have-
ImageFolder-data part didn't need to change):**
1. `preprocessing.py` — `index_dataset()` (Format A: ImageFolder-style,
   reused as-is; Format B annotation-file fallback kept for robustness but
   unused for PlantVillage), `select_top_classes()` (reused as-is).
2. `dataset.py` — renamed `PestImageDataset` → `LeafImageDataset`; kept
   ±15° rotation (not the 360° used in the Agriculture-Vision attempt,
   since leaves have a natural orientation unlike aerial tiles) — this is
   the one meaningful augmentation difference from the other two image
   projects, worth remembering if asked to explain augmentation choices.
3. `models.py`, `train.py`, `evaluate.py`, `interpret.py` — copied verbatim
   from `pest_project/`; fully domain-agnostic.

**Main deliverable:** `notebooks/plant_disease_detection.ipynb` — 9 sections
mapped 1:1 to report sections. Already executed once against synthetic data
(procedurally generated leaf-green images with colored lesion-like patches
per class, 5-class reduced-threshold test) — zero errors; Grad-CAM correctly
centered on the lesion patches rather than healthy leaf tissue. Notebook was
then rebuilt with the real-world defaults (`N_CLASSES=15`,
`MIN_PER_CLASS=200`) for actual handoff — this final version has NOT been
executed (thresholds don't fit the tiny synthetic test data), but the
identical code path was validated at smaller scale, so it should work
unchanged once real data (which comfortably clears both thresholds) is
in place.

**Also present:** `smoke_test.py` (generates synthetic leaf-lesion images,
runs the entire pipeline standalone — note it uses N_CLASSES=5,
min_per_class=10 internally, appropriate for its own small synthetic set),
`build_notebook.py` (regenerates the notebook programmatically — edit this
and rerun rather than hand-editing the `.ipynb`).

## Immediate Next Step

1. User downloads PlantVillage from a Kaggle mirror (~2GB) and places the
   class folders directly under `data/`.
2. Run `notebooks/plant_disease_detection.ipynb` top to bottom. Adjust
   `N_CLASSES` / `MIN_PER_CLASS` in Section 2 if the default 15-class scope
   needs tuning, or restrict to a single crop's classes for a more focused
   comparison (noted as an option in the notebook markdown).
3. Fill in Section 9 (Conclusion) using real metrics/Grad-CAM results,
   including the honest three-dataset retrospective already prompted there.
4. Adapt notebook markdown + figures into the actual 5–8 page report.

## Reference Material
`ML_Fundamentals_Reference.md` — full formula/definition reference from the
course's 145-slide deck, mapped to report sections. Applies unchanged across
all five project topics/datasets explored during this exam.
