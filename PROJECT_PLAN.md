# Predictive Maintenance — Engine Condition Classification — Project Plan
*Exam 3 — Own ML Project. Dataset: `engine_data.csv` (automotive engine sensor data, binary condition label)*

---

## 1. Problem Statement & Relevance

**Task:** Predict `Engine Condition` (binary: 0/1) from six live engine sensor readings — RPM, lubricating oil pressure, fuel pressure, coolant pressure, lubricating oil temperature, coolant temperature.

**Why this matters (technical/business/social context):**
This is the consumer/automotive counterpart to industrial predictive maintenance: rather than a factory floor with dozens of machine types, this is a single vehicle engine instrumented with the sensors already present on most modern cars (OBD-II style telemetry). The business case is the same shape as the industrial one but the stakeholders are different — a fleet operator, a car manufacturer's telematics service, or eventually an individual driver's dashboard warning light, rather than a factory maintenance scheduler. Catching an engine problem from sensor drift before it causes a breakdown or a safety incident (e.g. on a highway) is a directly actionable, safety-relevant use of a classification model — a false negative here (predicting "fine" when the engine is actually in trouble) is a more serious failure mode than a false positive, which matters directly for how we choose evaluation metrics (§6) and pick between using accuracy alone vs. class-sensitive metrics.

**Framing choice:** Binary classification — the label is already binary and, unlike the earlier RUL dataset, there's no leakage or derivation issue to resolve. This directly maps onto the reference material's §1.2 classification-loss section (binary cross-entropy + sigmoid output), giving a clean project fit for exactly the mechanics the exam's Theoretical Q1/Q2 ask about.

---

## 2. Dataset Description, Audit & Justification

**Source:** Automotive engine sensor dataset (Kaggle-family "engine health / predictive maintenance" dataset — precise original authorship unclear from the file alone; see label-semantics caveat below), 19,535 rows × 7 columns (6 features + label), one row per sensor snapshot.

**Why this dataset:** Small enough to make overfitting, regularization, and (optionally) data augmentation *genuinely load-bearing decisions* rather than formalities — unlike a 500k-row dataset where a model can hardly help but generalize. Also, critically, the features are only weakly correlated with the label individually (see audit below), which gives a neural network real room to demonstrate value over a linear baseline — the opposite situation from the factory dataset we scoped first, where one feature nearly solved the whole problem outright.

### 2.1 Data audit findings

| Check | Result | Consequence |
|---|---|---|
| Missing values | None across all 7 columns | No imputation needed |
| Duplicate rows | None | No deduplication needed |
| Physically impossible values | None — all pressure/temperature/RPM values are positive; no negative pressures or sub-zero absolute temperatures | No hard data-cleaning removals required |
| Class balance | 63% class 1 / 37% class 0 | Moderate imbalance — not extreme enough to require SMOTE, but real enough to justify class-weighted loss and to make accuracy-alone a misleading headline metric (§6) |
| Correlation of each feature with the label | All weak: `Engine rpm` strongest at **r = -0.27**; `Lub oil pressure` 0.06; `Fuel pressure` 0.12; `Coolant pressure` -0.02; `lub oil temp` -0.09; `Coolant temp` -0.05 | No single dominant feature (contrast with the factory dataset's 0.985-correlated `Operational_Hours`) — the label depends on a genuinely multivariate, nonlinear combination, which is exactly the setting where an MLP should outperform a linear baseline |
| Quick logistic-regression sanity baseline (StandardScaler + `LogisticRegression`, 80/20 stratified split) | **AUC 0.69, accuracy 66%**, recall on class 0 only 0.29 | Confirms a linear decision boundary badly under-serves the minority/problem class — establishes a concrete bar the MLP needs to clear, and motivates threshold/metric choices in §6 |
| Outlier scan (IQR rule) per column | `lub oil temp` flags 13.4% of rows, but its IQR is only 2.35 on a tight 71.3–89.6 range — this reflects a naturally narrow, dense distribution, **not data errors**; `Fuel pressure` 5.8%, `Coolant pressure` 4.0%, `Engine rpm` 2.4% similarly reflect natural spread | **No rows removed on IQR grounds** — the IQR rule is known to over-flag tightly-clustered, roughly-normal features, and none of the "outlier" values are physically implausible |
| Single extreme value: `Coolant temp` = 195.5°C (dataset range is otherwise ~61–90°C) | Exactly one row out of 19,535 | Kept — with only one occurrence it can't meaningfully bias training, and it's plausible as a genuine (rare) overheating event, which is precisely the kind of signal a predictive-maintenance model should be sensitive to, not discard |
| Label semantics | The source metadata does not explicitly document which integer (0 or 1) means "healthy" vs "faulty" | **Explicit assumption stated for the report:** treated as unconfirmed; results are reported symmetrically (both classes' precision/recall/F1 shown) so the analysis doesn't silently depend on getting this the "right" way round. Flagged as a data-provenance limitation in §2.2. |

**Report framing for §2 ("Data choice"):** lead with the audit table exactly as with the previous dataset — it demonstrates the same rigor, and the finding here ("weak individual correlations, but a linear baseline still gets meaningfully above chance, and there's real headroom for a nonlinear model") is a more compelling motivation for the project's model choice than the earlier dataset's near-deterministic single-feature story.

### 2.2 Expected limitations (for the report's Data Choice section)
- No documented label semantics or data-collection methodology (unclear whether this is real fleet telemetry, a lab bench rig, or itself partially synthetic) — treat performance figures as a demonstration of methodology rather than a claim about real-world deployability.
- Only 6 sensor channels — real automotive OBD-II systems expose more (e.g. throttle position, O2 sensor readings, battery voltage); a production system would likely have more signal available than this dataset does.
- 19,535 rows is workable but modest for a 6-feature problem — supports the case for regularization (§5.5) and for treating data augmentation as worth trying rather than dismissing outright (§3, §5.5).
- Single-snapshot rows again (no per-vehicle time series), so — as with the factory dataset — RNN/Transformer approaches aren't a natural fit for this particular file, despite being relevant to Theoretical Q1's broader comparison.

---

## 3. Preprocessing Pipeline

**Final feature set (6 features, all retained — no drops needed this time):**
`Engine rpm`, `Lub oil pressure`, `Fuel pressure`, `Coolant pressure`, `lub oil temp`, `Coolant temp` — all numeric, no categorical columns, no identifier column to strip.

**Target:** `Engine Condition` (binary, 0/1)

**Pipeline steps, in order:**
1. Load CSV. No column drops needed (audit found no leakage, no identifiers, no structurally-missing sensor columns this time).
2. **Split first**, before fitting any transformer: 70% train / 15% validation / 15% test, **stratified** on `Engine Condition` (important here, unlike the RUL project, because of the 63/37 imbalance — stratification keeps that ratio consistent across all three splits), `random_state=42`.
3. **Fit `StandardScaler` on the 6 numeric features using train data only**, transform train/val/test with the same fitted scaler. Necessary for the MLP (six features on very different native scales — RPM in the hundreds, pressures in single digits, temperatures in the 60s–90s — would otherwise distort gradient magnitudes).
4. **Class imbalance handling** — three complementary techniques considered, with a recommendation:
   - **Class-weighted loss** (weight the minority class 0 higher in `BCEWithLogitsLoss`'s `pos_weight`, or equivalently in `class_weight='balanced'` for the logistic regression baseline) — cheapest, no data duplication, directly addresses the recall problem the sanity-check baseline exposed. **Recommended as the primary strategy.**
   - **Random oversampling** of the minority class in the training set only (never in val/test, to avoid evaluating on duplicated rows) — considered as an ablation/comparison.
   - **SMOTE** (synthetic minority oversampling) — considered but likely unnecessary given the imbalance is moderate (63/37, not e.g. 95/5); adds complexity without a clear expected payoff here, so treated as optional/out of scope, with the reasoning stated explicitly in the report rather than silently omitted.
5. **Data augmentation — justified here, unlike the RUL project.** With ~19.5k rows (vs. the earlier dataset's 500k) and a real risk of the MLP memorizing rather than generalizing, **Gaussian noise jitter** on the scaled numeric features during training (add small random noise, e.g. σ = 0.05 in standardized units, to each feature on the fly per mini-batch) is included as a regularizer — simulating natural sensor measurement noise and discouraging the network from fitting exact training values. This is applied **only to the training set**, never to validation/test. Framed in the report as: "unlike the first dataset considered for this project (500k synthetic rows, effectively noise-dominated), this smaller, weak-signal dataset is a case where augmentation is actually justified, and its effect is measured directly via an ablation (with vs. without jitter) in §7."

---

## 4. Baseline Model

- **Baseline — Logistic Regression**, `class_weight='balanced'`, same scaled 6-feature input as the MLP. Justified by the same reference-material reasoning as before (§4 of the reference doc: linear models are simple, fast, interpretable — but here, unlike the RUL project, the audit's own sanity-check run already shows this baseline is clearly beatable (AUC 0.69, poor minority recall), giving the MLP a real, non-trivial bar to clear rather than an already-solved problem.
- (Optional stretch baseline: a single-feature logistic regression on `Engine rpm` alone, mirroring the RUL project's Baseline B, to show explicitly that no single feature comes close to sufficient — reinforcing the "genuinely multivariate" framing.)

---

## 5. Main Model: Feedforward Neural Network (MLP Classifier)

### 5.1 Why an MLP (not a CNN or RNN)
Same reasoning as the RUL project: this is tabular, non-sequential, non-spatial data (6 independent sensor readings per snapshot, no grid structure for a convolution and no temporal ordering across rows for a recurrent model). A fully-connected feedforward network remains the architecturally appropriate choice, restated here because the classification framing (rather than regression) changes the *output layer and loss*, not the *hidden architecture* rationale.

### 5.2 Architecture

```
Input (6 features, standardized)
  → Linear(6 → 32)  → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(32 → 16)  → BatchNorm1d → ReLU → Dropout(0.2)
  → Linear(16 → 8)                → ReLU
  → Linear(8 → 1)                                          [output: raw logit]
```

- **Much smaller than the RUL project's MLP** (32-16-8 vs 128-64-32) — deliberately, because the input space here is only 6 features (vs 49 after one-hot encoding), and the dataset is 25× smaller. Matching capacity to the actual complexity of the problem is itself a stated design decision, not an oversight — an oversized network here would be the textbook overfitting setup the reference material warns about (§8).
- **Output layer:** single linear unit producing a **raw logit**, with **sigmoid applied inside the loss function** (see §5.4) rather than as a separate network layer — this is the numerically stable standard PyTorch pattern (`BCEWithLogitsLoss` combines sigmoid + BCE in one numerically stable operation, avoiding the separate-sigmoid-then-BCE approach's potential floating-point issues near 0/1). At inference time, `torch.sigmoid(logit)` is applied explicitly to get the class-1 probability.

### 5.3 Activation function — ReLU (hidden layers), Sigmoid (output, implicit)
- **Hidden layers: ReLU**, `max(0, z)`, for the same reasons as the RUL project (§3 of reference material) — avoids vanishing-gradient saturation, computationally cheap, standard default.
- **Output: Sigmoid**, `σ(z) = 1/(1+e^{-z})` — this is the reference material's §1.2 explicit pairing for binary cross-entropy: "Pairs naturally with a sigmoid output activation." This is the one place this project's architecture differs structurally from the RUL project's (which correctly used *no* output activation, since RUL is an unbounded continuous value) — worth stating explicitly in the report as evidence of understanding *why* the two projects' output layers differ, not just describing each in isolation.

### 5.4 Loss function — Binary Cross-Entropy
$$BCE = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right]$$
Directly from reference material §1.2 — this is a 2-class problem, so binary (not categorical) cross-entropy applies. Implemented via `nn.BCEWithLogitsLoss(pos_weight=...)`, where `pos_weight` implements the class-imbalance weighting from §3 (up-weighting the minority class's contribution to the loss so the network isn't rewarded for simply predicting the majority class most of the time). MSE (used for the RUL project) is explicitly *inappropriate* here — the reference material is clear that MSE/MAE are regression losses and don't apply to a classification target, which is worth one contrastive sentence in the report tying back to the earlier project.

### 5.5 Regularization
| Method | Used here | Mechanism / justification |
|---|---|---|
| Dropout (0.3 / 0.2) | Yes, higher than the RUL project's (0.2/0.2/0.1) | Smaller dataset (19.5k vs 500k rows) means a materially higher overfitting risk per parameter, so stronger dropout is warranted |
| Batch Normalization | Yes, on the first two hidden layers | Stabilizes training, allows a higher learning rate |
| Class-weighted loss | Yes (`pos_weight` in `BCEWithLogitsLoss`) | Directly counteracts the 63/37 imbalance at the loss level, rather than only at the data level |
| Gaussian noise augmentation (train-only) | Yes, as an explicit ablation (§3, §7) | Regularizes against memorization on a comparatively small dataset — see §3 for why this is justified here but wasn't for the RUL project |
| Weight decay (L2) | Yes, small value (e.g. 1e-4 — slightly higher than the RUL project's 1e-5, again reflecting the smaller dataset) | Additional smoothness prior |
| Early stopping | Yes — monitor validation BCE loss (or validation F1, given the imbalance — F1 arguably more decision-relevant than raw loss here), patience-based, restore best checkpoint | Same textbook overfitting-signature countermeasure as the RUL project (reference material §8) |

### 5.6 Optimizer — Adam
Same justification as the RUL project (§5.6 of that plan): per-parameter adaptive learning rates via momentum + RMSProp-style scaling, standard default for feedforward nets, handles the six differently-scaled (even after StandardScaler, batch-to-batch gradient noise differs across features) input dimensions well. `torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)`.

### 5.7 Gradient descent variant — Mini-batch
Same reasoning as before (reference material §2.3), though at this dataset size (≈13,700 training rows after the 70% split) the batch-size choice matters less dramatically than at 350,000 rows — batch size 32 or 64 is a reasonable default given the smaller dataset (smaller batches per epoch = more gradient updates per epoch, which can help on a smaller dataset with more room for iterative refinement without excessive computational cost).

### 5.8 Learning rate
Start at `1e-3` (Adam default), same reasoning as the RUL project §5.8 (too small → slow convergence; too large → risk of overshoot/divergence, compounded here by the smaller batch-to-batch sample size making gradient estimates noisier).

---

## 6. Evaluation Metrics

Per reference material §9, and *unlike* the RUL project, classification metrics now apply directly:
- **Accuracy** — reported, but explicitly **not** the headline metric, given the 63/37 imbalance (a model predicting "class 1" for everything would already score 63% accuracy while being useless — the reference material's own caution about single-metric overconfidence, echoed here in classification form rather than the R² form it took in the RUL project).
- **Precision, Recall, F1 — per class**, not just macro-averaged, so the minority-class (whichever class turns out to be the "problem" engine state, per the labeling caveat in §2.1) performance is visible on its own, not washed out by the majority class's easier performance.
- **Confusion matrix** — to see the actual false-negative/false-positive trade-off directly, which matters given the safety framing in §1 (a missed "faulty" prediction is worse than a false alarm).
- **ROC-AUC** — threshold-independent summary, useful for comparing the logistic regression baseline against the MLP on equal footing regardless of where each model's decision threshold ends up.

---

## 7. Failure-Case & Interpretation Plan

1. **Baseline comparison table**: Logistic Regression vs MLP, on accuracy / per-class precision-recall-F1 / ROC-AUC — the central table, directly showing whether the nonlinear model earns its complexity (expected: yes, given the audit's weak-individual-correlation, low-linear-baseline-AUC finding).
2. **Augmentation ablation**: MLP with vs. without Gaussian noise jitter during training — validates (or disproves) the §3/§5.5 justification empirically rather than asserting it.
3. **Confusion matrix walkthrough**: which class does the model confuse most, and does that align with the label-semantics uncertainty flagged in §2.1?
4. **Feature importance** (permutation importance on the trained MLP): does `Engine rpm` (the strongest individual correlate) dominate the trained model's decisions the way `Operational_Hours` dominated the RUL project, or does the MLP genuinely spread importance across multiple features — this is the key contrastive finding between the two datasets worth writing up explicitly.
5. **Explicit limitation statement**: unresolved label-semantics ambiguity (§2.1) and the modest dataset size relative to real fleet telemetry systems.

---

## 8. Report Structure Mapping (5–8 pages)

| Report section | Plan section(s) to draw from |
|---|---|
| Problem relevance | §1 |
| Data choice (source, size, quality, limitations) | §2 |
| Model setup justification (preprocessing + architecture) | §3, §5.1–§5.3 |
| Why BCE loss / sigmoid / ReLU / Adam / mini-batch / class weighting (theory tie-in) | §5.4–§5.8 (also directly answers Theoretical Q2) |
| Baselines | §4 |
| Evaluation | §6, §7.1–§7.3 |
| Interpretation | §7.4–§7.5 |
| Conclusion: "why this data/setup were the right choice (or not)" | Synthesize §2.1 audit + §7.1/§7.4 findings — likely conclusion: a stronger fit than the initially-scoped synthetic factory dataset specifically *because* the linear baseline is clearly beatable, giving the neural network a meaningful role to demonstrate, at the cost of thinner data-provenance documentation |
| Theoretical Q1 (linear/CNN/RNN comparison) | Reference doc §4 directly — §5.1's "why MLP not CNN/RNN" reasoning is the concrete anchor, and this project's contrast with the RUL project (regression, no output activation) vs this one (classification, sigmoid output) is a good worked example for discussing linear-model / classification-vs-regression nuance |
| Theoretical Q2 (gradient descent) | Reference doc §2 directly — §5.6–§5.8 give a second worked example (smaller batch size, higher weight decay, class-weighted loss) to contrast against the RUL project's |

---

## 9. Implementation Stack & File Structure

**Stack:** Python, pandas, scikit-learn (`StandardScaler`, `LogisticRegression`, `train_test_split`, `permutation_importance`, classification metrics), PyTorch (MLP classifier + training loop), matplotlib/seaborn (plots). *(PyTorch still needs installing: `pip install torch --break-system-packages`.)*

```
engine_condition_project/
├── data/
│   └── engine_data.csv
├── src/
│   ├── preprocess.py      # §3 pipeline: split, scale, class-weight setup, augmentation fn
│   ├── baselines.py       # §4 logistic regression baseline(s)
│   ├── model.py           # §5 MLP classifier (nn.Module)
│   ├── train.py           # training loop, early stopping, checkpointing, augmentation ablation switch
│   └── evaluate.py        # §6-§7 metrics, confusion matrix, permutation importance
├── notebooks/
│   └── main_analysis.ipynb  # end-to-end reproducible notebook, the primary deliverable
├── outputs/
│   ├── figures/            # confusion matrix, ROC curve, importance chart, training curves
│   └── metrics.json
└── report/
    └── exam3_report.docx    # 5-8 page final report
```

---

## Open items to confirm before coding begins
- MLP hidden-layer sizes (32-16-8) and dropout rates are a reasonable starting point, to be tuned against validation F1/loss once training is running.
- Batch size (32 vs 64) and early-stopping patience to be finalized empirically.
- Whether to include the optional single-feature baseline (`Engine rpm` only) alongside the full logistic regression baseline — cheap to add, your call once we're implementing.
