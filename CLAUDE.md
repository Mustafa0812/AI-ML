# CLAUDE.md — Engine Condition Predictive Maintenance Project

**Exam 3 — Own ML Project.** This file is the handoff context for continuing this project in Claude Code. Read this fully before making changes — several design decisions here were made after explicitly rejecting simpler alternatives, and re-litigating them wastes time and risks contradicting the report already being drafted around these results.

---

## Project summary

Binary classification: predict `Engine Condition` (0/1) from 6 automotive engine sensor readings (RPM, lub oil pressure, fuel pressure, coolant pressure, lub oil temp, coolant temp). Dataset: `data/engine_data.csv`, 19,535 rows, no missing values, no duplicates, 63/37 class balance.

**Full design rationale lives in `PROJECT_PLAN.md`** — read that first for the "why" behind every choice below. This file is state/status, not justification.

## Dataset history (important context, don't re-suggest these)

This project went through two dataset iterations before settling here:
1. **First dataset** (500k-row synthetic factory sensor sim, `Remaining_Useful_Life_days` regression target) — rejected/superseded. Found `Operational_Hours` correlated at r=-0.985 with RUL, meaning one feature nearly solved the whole task — not a compelling demonstration of neural net value. Also found `Failure_Within_7_Days` was a leaky derivative of RUL (RUL≤7 → failure=True in 99.9% of rows).
2. **Current dataset** (`engine_data.csv`, this file) — chosen because weak individual feature correlations (all |r|<0.27) leave genuine room for a nonlinear model to add value over a linear baseline, and the smaller size (19.5k rows) makes regularization/augmentation load-bearing decisions rather than formalities.

**Do not suggest going back to the factory dataset or switching again** unless the user explicitly raises it — this was already discussed at length and settled.

## Key findings already established (cite these, don't re-derive)

- **Label semantics unconfirmed**: no documentation found (searched) for whether `Engine Condition=0` means healthy or faulty. Report treats this as an explicit limitation and shows both classes' metrics symmetrically rather than assuming.
- **No outlier removal performed**: IQR flags on `lub oil temp` (13.4%) etc. reflect naturally tight distributions, not data errors — verified no physically impossible values (no negative pressures, etc.) exist in the data.
- **Performance ceiling confirmed at AUC ≈ 0.70**, verified across 5 different model families:
  - Logistic Regression (all features): AUC 0.696
  - Logistic Regression (Engine rpm only): AUC 0.669
  - Random Forest (300 trees): AUC 0.685
  - Gradient Boosting (300 trees): AUC 0.700
  - **MLP (this project's main model, no augmentation): AUC 0.701** ← best, but only marginally
  - Gradient Boosting + 6 engineered interaction/ratio features: AUC 0.700 (no improvement — ceiling holds even with feature engineering)
  - **Conclusion for the report: this is a data information ceiling, not a modeling limitation.** State this plainly; don't chase further architecture tuning trying to "beat" ~0.70 AUC — the diagnostic work above already demonstrates why that's not productive.
- **Augmentation ablation result**: Gaussian jitter (σ=0.05, train-only) *slightly hurt* performance (AUC 0.699 vs 0.701 without). Report this honestly as a negative result — likely because dropout+batchnorm+weight_decay already sufficiently regularize a 993-parameter model on 13.6k training rows.
- **Permutation importance**: `Engine rpm` dominates (mean AUC drop 0.138 when shuffled) — 4x the next feature (`Fuel pressure`, 0.033). Consistent with the correlation audit.
- **Failure case**: MLP accuracy drops sharply as Engine rpm increases — 78.9% in the lowest RPM quartile down to 54.8% in the highest.
- **MLP vs Logistic Regression trade-off** (from confusion matrices, test set n=2931): MLP has notably better class-0 recall (72.9% vs 60.5%) but worse class-1 recall (58.5% vs 67.8%). Whether this trade-off is "good" depends on the unresolved label-semantics question above.

## Implementation status

All core modules built and tested working in `src/`:

| File | Status | Purpose |
|---|---|---|
| `src/preprocess.py` | ✅ Done, tested | Load, stratified 70/15/15 split, StandardScaler (train-fit only), `pos_weight` computation, `add_gaussian_jitter()` augmentation fn |
| `src/model.py` | ✅ Done, tested | `EngineConditionMLP`: 6→32→16→8→1, BatchNorm+ReLU+Dropout(0.3/0.2), raw logit output. 993 params. |
| `src/baselines.py` | ✅ Done, tested | Full-feature and single-feature (`Engine rpm` only) logistic regression, `class_weight='balanced'` |
| `src/train.py` | ✅ Done, tested | Mini-batch training loop, Adam (lr=1e-3, weight_decay=1e-4), `BCEWithLogitsLoss(pos_weight=...)`, early stopping (patience=15) on val loss, `use_augmentation` toggle |
| `src/evaluate.py` | ✅ Done, tested, run end-to-end | Full comparison (both baselines + both MLP variants), permutation importance, residual-by-RPM-band analysis, all plots, saves `outputs/metrics.json` |

**Reproduce all results:** `cd src && python3 evaluate.py` — takes ~1-2 min, regenerates everything in `outputs/`.

Figures already generated in `outputs/figures/`: `confusion_matrices.png`, `roc_curves.png`, `training_curves.png`, `permutation_importance.png`. Metrics in `outputs/metrics.json`.

## Remaining work (not yet started)

1. **`notebooks/main_analysis.ipynb`** — the primary reproducible deliverable. Should walk through the full pipeline end-to-end (audit → preprocess → baselines → MLP training → augmentation ablation → evaluation → interpretation) as a narrative notebook, not just import the `src/` modules silently — the exam wants to see the reasoning, not just results. Reuse the `src/` modules as the underlying implementation (`sys.path.insert(0, '../src')` then import) rather than duplicating code, but write markdown cells that explain each step referencing `PROJECT_PLAN.md`'s justifications.
2. **`report/exam3_report.docx`** — 5-8 page report. Structure mapping is in `PROJECT_PLAN.md` §8. Use the `docx` skill (`/mnt/skills/public/docx/SKILL.md`) when generating this. Must include: problem relevance, data choice + audit table, model setup justification (loss/activation/optimizer reasoning from `PROJECT_PLAN.md` §5.3-§5.8), baseline comparison, evaluation (all 4+ models table, confusion matrices, ROC), interpretation (permutation importance, RPM-band failure case, augmentation ablation as a negative result), and the conclusion synthesizing the AUC-ceiling finding honestly.
3. **Theoretical Q1 & Q2 answers** — separate from the project report (see `exam_3_own_project.pdf` for exact wording). Reference material for these is `ML_Fundamentals_Reference.md` (already in project context, not yet copied into this project folder — copy it in if needed). `PROJECT_PLAN.md` §8's table maps which sections answer which theoretical sub-question, with this project's own architecture choices (MLP not CNN/RNN, sigmoid+BCE not linear+MSE) as concrete worked examples to fold in.

## Conventions to maintain

- Random seed `42` everywhere (splits, model init, DataLoader shuffling) — reproducibility matters for the "reproducible code/notebook" deliverable requirement.
- Never fit `StandardScaler` or anything else on val/test data — train-only fitting is already correctly implemented in `preprocess.py`, don't regress this.
- Report both classes' precision/recall/F1 individually, never just macro-averaged accuracy — the imbalance and label-semantics ambiguity make this important.
- When discussing results in the report/notebook, don't oversell the AUC ceiling finding as a project failure — it's framed as a rigor demonstration throughout `PROJECT_PLAN.md` and should stay that way for consistency.
