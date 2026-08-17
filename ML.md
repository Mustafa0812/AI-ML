# Exam 3 — Own ML Project: Bearing Fault Classification

## Context
University "Exam 3" deliverable. Two theoretical questions (independent of the project — already covered separately, see `ML_Fundamentals_Reference.md`) plus a hands-on ML project.

**Deliverables required:**
- 5–8 page report
- Reproducible code/notebook
- Short conclusion: "Why this data and setup were the right choice (or not)"

**Report must cover:** problem relevance, data choice justification (source/size/quality/limitations), model setup justification (architecture + preprocessing), a baseline comparison, evaluation (metrics + failure cases), and interpretation (what the model learned, where it fails).

## Project Decision

**Topic: Bearing fault classification — CWRU Bearing Dataset**

Chosen over drone flight-anomaly detection and autonomous-driving behavioral cloning (rejected: too close to Mustafa's existing PilotNet-style project, weak on originality). Predictive maintenance won because it's the cleanest fit for mechatronics relevance, has a small well-documented benchmark dataset, and gives a strong baseline-vs-model comparison story.

**Constraint clarified: "pure ML model"** = no hand-crafted signal-processing features (no BPFO/BPFI-style feature engineering). Model must learn directly from raw signal / spectrogram — NOT a restriction against deep learning (CNNs are explicitly in scope; course's own Theoretical Q1 treats linear models / CNNs / RNN-Transformers as one family).

## Technical Plan

**Task:** 4-class classification — Normal / Inner race fault / Outer race fault / Ball fault — from short windows of vibration accelerometer signal.

**Dataset:** CWRU Bearing Data Center (Case Western Reserve University)
- Vibration signals, 12kHz/48kHz sampling, `.mat` files (readable via `scipy.io.loadmat`)
- Faults seeded via EDM at 3 locations, multiple severities (0.007"/0.014"/0.021"), 4 motor loads (0–3 hp)
- Status: **not yet downloaded**. Not accessible via Kaggle or the official Case Western host from this sandbox (network allowlist blocks both). GitHub mirror search was in progress and hit rate limits (429s from unauthenticated GitHub API/codeload requests). User has Kaggle access and may download+upload the dataset directly, or a GitHub mirror can be retried (candidates found: `s-whynot/CWRU-dataset`, `srigas/CWRU_Bearing_NumPy`, `shayanMoodi/CWRU_BearingDataset`, `LGDiMaggio/CWRU-bearing-fault-classification-ML`).

**Pipeline:**
1. Segment raw signal into fixed windows (e.g. 2048 samples), z-score normalize
2. Convert windows to spectrograms via STFT (generic transform, not domain feature engineering — allowed under "pure ML")
3. Baseline: logistic regression on flattened raw windows (deliberately weak, no feature engineering)
4. Main model: 2D-CNN on spectrograms — conv+pool blocks → FC → softmax over 4 classes
5. Optional stretch: LSTM/GRU on raw time series for a 3-model comparison

**Loss/training choices (justified from course material — see reference doc):**
- Categorical cross-entropy + softmax output (standard for multi-class classification)
- Mini-batch gradient descent (Adam optimizer — note: Adam isn't in the course slides, it's a standard extension of mini-batch GD with momentum + adaptive learning rates, worth flagging as outside-the-deck knowledge)
- Regularization: dropout and/or batch normalization inside the CNN to counter overfitting

**Evaluation:** accuracy, per-class F1, confusion matrix. Planned experiments: class-confusion analysis (e.g. early-stage inner-race fault vs. normal) and cross-load generalization (train on 1–2 hp, test on 3 hp).

**Interpretation:** Grad-CAM-style saliency on CNN spectrogram inputs, ideally checked against known bearing fault frequencies (BPFO/BPFI) post-hoc — model should discover these patterns without being told them, which is the intended "what did the model learn" narrative for the report.

## Reference Material
A full formula/definition reference (`ML_Fundamentals_Reference.md`, already generated) extracts everything relevant from the course's 145-slide deck: loss functions (MSE/MAE/RMSE/R²/BCE/cross-entropy), gradient descent (batch/SGD/mini-batch, learning rate, ascent vs. descent), CNN/RNN/LSTM/GRU/Transformer formulas, regularization methods, and evaluation metrics — mapped directly to which report section each piece supports.

## Immediate Next Step
Get the CWRU dataset into the working environment (user download+upload from Kaggle is the most reliable path given sandbox network restrictions), then start the preprocessing + baseline + CNN build.
