# ML Fundamentals Reference — Exam 3
*Extracted from Dr.-Ing. Grigory Devadze's "Artificial Intelligence and Machine Learning" course slides. Organized for direct use in Theoretical Q1, Theoretical Q2, and the CWRU bearing fault classification project report.*

---

## 1. Loss Functions

### 1.1 Regression losses (relevant for baselines / comparison, not your project directly)

**Mean Absolute Error (MAE)**
$$MAE = \frac{1}{m}\sum_{i=1}^{m}|y_i - \hat{y}_i|$$
Average absolute deviation. Lower = more accurate.

**Mean Squared Error (MSE)**
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$
Squares errors, so larger errors are penalized disproportionately more than MAE.

**Root Mean Squared Error (RMSE)**
$$RMSE = \sqrt{MSE} = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat{y}_i)^2}$$
Same unit as the target variable — more interpretable than MSE.

**Coefficient of Determination (R²)**
$$R^2 = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$
- R² = 1 → perfect model
- R² ≈ 0 → model explains almost nothing
- R² < 0 → worse than predicting the mean
- **Caution flagged in the slides: an unusually high R² can itself be a sign of overfitting** — useful line for your Interpretation section.

### 1.2 Classification losses (**this is what your project uses**)

**Binary Cross-Entropy (BCE)** — for 2-class problems:
$$BCE = -\frac{1}{N}\sum_{i=1}^{N}\left[y_i \log(p_i) + (1-y_i)\log(1-p_i)\right]$$
Pairs naturally with a **sigmoid** output activation.

**Categorical Cross-Entropy** — for your **4-class** (Normal / Inner race / Outer race / Ball) problem:
$$\text{Cross-Entropy} = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{C} y_{ij} \cdot \log(p_{ij})$$
where $y_{ij}$ is the one-hot true label, $p_{ij}$ the predicted probability for class $j$, $N$ = samples, $C$ = classes.

**Why this is your loss function (write this directly into "Model setup justification"):**
- Your task is multi-class classification (4 fault categories), not regression — MSE/MAE don't apply.
- Cross-entropy pairs naturally with a **softmax** output layer, which is what you'll use to turn your CNN's final layer into class probabilities.
- It "penalizes large deviations between prediction and true label strongly" and is the standard, well-behaved loss for this exact setup.
- PyTorch usage (from the slides):
```python
criterion = nn.CrossEntropyLoss()
loss = criterion(predictions, labels)
```

---

## 2. Gradient Descent (Theoretical Q2)

### 2.1 Core idea
Gradient descent finds the parameters that minimize a loss/cost function by iteratively stepping in the direction that reduces the loss fastest — the negative gradient direction.

For linear regression (illustrative case from the slides), cost function:
$$MSE = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$$

Partial derivatives (needed to know which direction reduces error):
$$\frac{\partial}{\partial \theta_1}MSE = -\frac{2}{n}\sum_{i=1}^n x_i(y_i-\hat{y}_i)$$
$$\frac{\partial}{\partial \theta_0}MSE = -\frac{2}{n}\sum_{i=1}^n (y_i-\hat{y}_i)$$

**Update rule:**
$$\theta_1 := \theta_1 - \eta \cdot \frac{\partial}{\partial \theta_1}MSE$$
$$\theta_0 := \theta_0 - \eta \cdot \frac{\partial}{\partial \theta_0}MSE$$

For a general neural network, the same idea applies to every weight $w$ using backpropagation to compute $\partial L/\partial w$ across all layers.

### 2.2 Role of the loss function and its gradient
- The loss function quantifies how wrong the model's predictions are.
- The gradient (vector of partial derivatives w.r.t. every parameter) tells you the direction of steepest **increase** in loss.
- Gradient descent moves parameters in the **opposite** direction to decrease loss.

### 2.3 Batch vs. mini-batch vs. stochastic gradient descent

| Method | Computes gradient on... | Advantages | Disadvantages |
|---|---|---|---|
| **Batch GD (BGD)** | All data points | Very stable, accurate convergence | Computationally expensive for large datasets |
| **Stochastic GD (SGD)** | One random data point per step | Very efficient for large datasets | Fluctuates strongly (high variance) |
| **Mini-Batch GD (MBGD)** | Small group (e.g. 32/64 points) | Balance of stability and efficiency | Requires careful batch-size tuning |

Visually (from the slides): batch GD walks smoothly toward the minimum on the loss surface; SGD zigzags/jumps around; mini-batch sits in between.

**Relevance to your project:** you'll almost certainly train with mini-batch gradient descent (standard for CNNs) — this is worth one sentence in your Model Setup section, e.g. "mini-batch gradient descent (via the Adam optimizer, batch size 32) was chosen for its balance of training stability and computational efficiency on the windowed vibration dataset."

### 2.4 Learning rate ($\eta$)
Controls the step size of each update.
- **Too small** → many tiny steps, training takes very long to converge.
- **Too large** → overshoots the minimum, can diverge entirely instead of converging.
- Not every loss surface is a clean bowl — local minima and plateaus can trap gradient descent, which is part of why the choice of learning rate (and optimizer) matters.

### 2.5 Gradient descent vs. gradient ascent
- **Gradient descent**: moves parameters in the direction that **minimizes** a loss function (most supervised learning — minimizing prediction error).
- **Gradient ascent**: moves parameters in the direction that **maximizes** an objective function — used when you want to increase something rather than decrease it, e.g. maximizing expected reward in reinforcement learning, or maximizing a likelihood function directly (rather than minimizing negative log-likelihood). In practice, gradient ascent on an objective $J$ is mathematically identical to gradient descent on $-J$, so frameworks usually just implement descent and let you flip the sign.

---

## 3. Neural Network Fundamentals

**Neuron computation** (used identically in every layer type):
$$z = \sum_i w_i x_i + b \quad \text{(preactivation)}$$
followed by a non-linear activation function applied to $z$.

**Layer as matrix operation:**
$$Z = WX + b$$

**Why non-linearities are necessary:** stacking purely linear transformations collapses to a single linear function no matter how many layers you add — non-linear activations are what let a network approximate complex, non-linear relationships.

**ReLU** (used almost universally in modern deep learning):
$$f(z) = \max(0, z)$$
Simple, fast, sets negative values to zero, avoids some of the saturation problems of sigmoid.

**Sigmoid / Tanh:** older activations, prone to saturation (vanishing gradients) for large |z|; sigmoid still used for binary-classification output layers, tanh appears inside RNN/LSTM cells.

---

## 4. Linear Models, CNNs, RNN/Transformers (Theoretical Q1 — comparison table)

| | **Linear models** | **CNNs** | **RNN / Transformers** |
|---|---|---|---|
| **Typical data** | Tabular/structured, roughly linear relationships | Images, audio, grid-like/spatially-local data (also adapted for text) | Sequential data: time series, text, speech |
| **Strengths** | Simple, interpretable, fast, exact analytical solution possible for regression | Sparse/local connections → far fewer parameters than fully connected; translation invariance — detects patterns anywhere in input | Memory of previous steps (RNN); Transformers give direct access between any two positions, fully parallelizable, no vanishing-gradient bottleneck |
| **Weaknesses** | Cannot capture non-linear/complex relationships | Less natural fit for irregular/non-grid data | RNNs: sequential computation (no parallelism), vanishing/exploding gradients on long sequences; Transformers: computationally heavier, need lots of data |
| **Overfitting risk & regularization** | Low model capacity = lower overfitting risk typically; regularize via fewer features, L2/weight decay | Moderate–high capacity; regularized via dropout, batch normalization, data augmentation, weight decay | High capacity, especially Transformers; regularized via dropout, layer normalization, early stopping, weight decay |
| **Example use case (from slides)** | Sales/revenue forecasting from advertising budget (multiple linear regression) | Image recognition (ResNet, VGG, AlexNet), speech synthesis (WaveNet) | Machine translation, stock market/time-series forecasting; Transformers → LLMs (GPT, BERT) |

**Your project use case:** vibration-signal fault classification is a CNN application on spectrograms — directly analogous to the "image recognition" use case above, just with a spectrogram instead of a photo.

---

## 5. CNNs (detail)

- Specialized for **local connections** of features; convolutions detect local correlations in the data.
- **Reduced parameter count** vs. fully-connected layers — filters are shared/sparse rather than densely connecting every input to every neuron.
- **Translation invariance** — a learned filter detects its pattern regardless of where it appears in the input. This is exactly why a CNN can find a fault signature in a spectrogram whether it occurs early or late in the window.
- Typical structure: convolution → activation (ReLU) → pooling, stacked in blocks, then a fully-connected classification head with softmax output.

---

## 6. RNNs, LSTM, GRU (detail — for Theoretical Q1's RNN family, useful background even though your project uses a CNN)

**Vanilla RNN cell:**
$$h_t = \tanh(W_h \cdot h_{t-1} + W_x \cdot x_t + b)$$
$$y_t = W_y \cdot h_t + b_y$$

**Vanishing / exploding gradients:** Backpropagation Through Time (BPTT) multiplies gradient terms across every time step:
$$\frac{\partial L}{\partial h_k} = \frac{\partial L}{\partial h_T}\prod_{i=k+1}^{T}\frac{\partial h_i}{\partial h_{i-1}}$$
Whether this vanishes or explodes depends on the largest eigenvalue $\lambda_{max}$ of the recurrent weight matrix $W_h$:
- $\|\lambda_{max}\| \ll 1$ → gradient vanishes exponentially, early time steps stop influencing the loss
- $\|\lambda_{max}\| \approx 1$ → stable (ideal, hard to hit)
- $\|\lambda_{max}\| \gg 1$ → gradient explodes, training diverges

**Fixes:**
| Problem | Solution | Mechanism |
|---|---|---|
| Exploding | Gradient clipping | Cap gradient norm at a threshold |
| Vanishing | LSTM / GRU | Additive cell-state updates instead of multiplicative |
| Vanishing | Residual connections | Gradient flows directly through skip connections |
| Both | Batch/Layer normalization | Stabilizes activations |
| Both | Transformer | No recurrence — direct access to all positions |

**LSTM gate equations:**
$$f_t = \sigma(W_f\cdot[h_{t-1},x_t]+b_f) \quad \text{(forget gate)}$$
$$i_t = \sigma(W_i\cdot[h_{t-1},x_t]+b_i) \quad \text{(input gate)}$$
$$\tilde{C}_t = \tanh(W_C\cdot[h_{t-1},x_t]+b_C) \quad \text{(candidate cell state)}$$
$$C_t = f_t \cdot C_{t-1} + i_t \cdot \tilde{C}_t \quad \text{(cell state update)}$$
$$o_t = \sigma(W_o\cdot[h_{t-1},x_t]+b_o) \quad \text{(output gate)}$$
$$h_t = o_t \cdot \tanh(C_t) \quad \text{(hidden state)}$$

**GRU (simplified, 2-gate alternative):**
$$r_t=\sigma(W_r[h_{t-1},x_t]) \quad z_t=\sigma(W_z[h_{t-1},x_t])$$
$$\tilde{h}_t=\tanh(W[r_t\odot h_{t-1},x_t]) \quad h_t=(1-z_t)\odot h_{t-1}+z_t\odot \tilde{h}_t$$
Fewer parameters, faster training, comparable performance to LSTM in many cases. Rule of thumb from the slides: try GRU first, switch to LSTM if more capacity is needed.

---

## 7. Transformers (detail)

**Scaled dot-product attention:**
$$\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- Q (query) = "what am I looking for?"; K (key) = "what do I offer?"; V (value) = "what information do I provide?"; $d_k$ scales down overly large dot products.

**Multi-head attention:**
$$\text{MultiHead} = \text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O$$
Each head learns a different attention pattern (e.g. one local, one global). Original Transformer used $h=8$ heads.

**Architecture:** Encoder (multi-head self-attention + feed-forward, residual + layer norm after each sublayer) and Decoder (masked self-attention + attention over encoder output + feed-forward). Fully parallelizable — the key advantage over RNNs.

---

## 8. Regularization (directly relevant — your CNN will need this)

| Method | Mechanism |
|---|---|
| **Dropout** | Randomly deactivates neurons during training with probability $p$ (e.g. 0.1–0.5); forces redundant representations so no single neuron dominates; all neurons active again at inference |
| **Batch Normalization** | Normalizes activations per mini-batch (mean 0, variance 1), then rescales with learnable $\gamma,\beta$; stabilizes/accelerates training, allows higher learning rates |
| **Layer Normalization** | Normalizes across features instead of batch — standard for sequence models (RNNs, Transformers) |
| **Weight Decay** | L2 regularization of the weights |
| **Early Stopping** | Stop training once validation error starts rising while training error keeps falling — the classic overfitting signature |

**Batch normalization formula:**
$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2+\epsilon}} \qquad y_i = \gamma\hat{x}_i+\beta$$
where $\mu_B,\sigma_B^2$ are the mini-batch mean/variance, and $\epsilon$ prevents division by zero.

**For your report:** state plainly that deep networks tend to overfit (training error keeps falling while validation error rises) and that you're using dropout and/or batch normalization inside your CNN as the countermeasure — this is a direct, citable justification straight from course material.

---

## 9. Evaluation Metrics for Classification

**Accuracy:**
$$\text{Accuracy} = \frac{TP+TN}{TP+FP+TN+FN}$$

Classification framed formally: learn a mapping $f:\mathbb{R}^n \to \{C_1,\dots,C_k\}$ from a training set $D=\{(x_i,y_i)\}_{i=1}^m$. In the probabilistic view, the model approximates $P(y=C_k\mid x)$ and predicts:
$$\hat{y} = \arg\max_{C_k} P(y=C_k\mid x)$$

This is exactly what your softmax output layer does: outputs a probability per class, and you take the argmax as the predicted fault type.

**Note for your Evaluation section:** the slides explicitly warn that accuracy alone can mislead (analogous to the R² overfitting warning above) — pair it with **per-class F1** and a **confusion matrix**, which you already planned, to properly discuss failure cases (e.g., early-stage inner-race fault confused with normal).

---

## 10. Quick-Reference: How This Maps to Your Report

| Report section | What to cite from this doc |
|---|---|
| Model setup justification | §1.2 (why cross-entropy + softmax), §5 (why CNN for spectrograms), §8 (dropout/batch norm to control overfitting) |
| Baseline justification | §4 table (linear model strengths/weaknesses vs. CNN) |
| Training details | §2 (gradient descent variant, learning rate reasoning) |
| Evaluation | §9 (accuracy formula, why accuracy alone isn't enough) |
| Interpretation / failure cases | §1.1 R² overfitting warning (analogous logic for high train accuracy), §8 early stopping / validation-error divergence as the textbook overfitting signature |
| Theoretical Q1 | §4 comparison table directly |
| Theoretical Q2 | §2 in full (all four sub-parts) |
