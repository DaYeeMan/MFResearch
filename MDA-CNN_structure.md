# MDA-CNN — Structure Summary

**Paper:** *Multi-fidelity Data Aggregation using Convolutional Neural Networks* (Chen, Gao, Liu)  
**Summary produced from:** uploaded manuscript (Manuscriptdraftmultifidelity.pdf).

---

## 1. High-level overview
MDA‑CNN (Multi‑fidelity Data Aggregation using Convolutional Neural Networks) is a neural‑network framework that treats multi‑fidelity data as image‑like input tables and applies convolutional operations to capture relationships between any high‑fidelity (HF) datum and the **entire** set (or local neighborhoods) of low‑fidelity (LF) data. It is composed of three main components: multi‑fidelity data compiling, multi‑fidelity perceptive field & convolution, and a deep neural network for linear/nonlinear mapping.

---

## 2. Components & data flow

### 2.1 Multi‑fidelity data compiling (Input table)
- For each HF sample \(y_{H,i}\) the model builds an **input table** with `NL` rows (one per LF datum).
- Typical columns (bi‑fidelity, 1‑D input) are:
  - `y_L` (locations of LF samples)
  - `Q_L(y_L)` (LF outputs at those locations)
  - `y_{H,i}` (the HF location being predicted — same for every row)
  - `Q_L(y_{H,i})` (LF output at the HF input — repeated down the rows)
- Thus every HF target yields an input matrix shaped roughly `NL × C` (C = number of columns/features). The full set of HF targets yields `N_H` such tables as training input. fileciteturn1file6

### 2.2 Multi‑fidelity perceptive field and convolution
- The input table is treated analogously to an image. A **local receptive field** (a rectangular window over rows × columns) slides down the table (sliding by one row) — analogous to 1‑D convolution along the LF samples axis.
- Each receptive field is connected to a hidden neuron; applying the windows across the table constructs a **feature map** that detects one type of localized relationship between LF and the current HF datum.
- Multiple feature maps (i.e., multiple convolutional kernels / filters) are used to capture different localized relationship types. The paper emphasizes that **sufficient number of feature maps** is necessary (e.g., 64 feature maps used in several experiments). fileciteturn1file12turn1file15

### 2.3 Deep neural network for mapping
- Outputs of the convolutional layer (feature maps) are flattened/combined and fed into a fully connected deep neural network.
- The mapping inside the deep NN is decomposed into **linear** and **nonlinear** components:
  - A skip connection provides a linear path (learned linear mapping).
  - Fully connected layers with nonlinear activations capture residual/nonlinear mapping.
  - The two components are added before the final output (ResNet‑style residual formulation) to ease optimization. fileciteturn1file3

---

## 3. Architectural choices reported in the paper (typical settings)
- Convolutional layer:
  - **Number of feature maps (filters):** e.g., *64* in many applications (the paper also explores fewer maps like 3 and shows degraded performance). fileciteturn1file15
  - **Kernel (window) width:** small integer window along rows (e.g., 3 in some examples). fileciteturn1file11
  - *Only one convolutional layer was used in the paper's experiments; no pooling layers were applied (authors note that for higher‑dim / more complex data additional conv/pooling layers could be used).* fileciteturn1file1
- Fully connected network (after conv):
  - **Hidden layers:** 2 hidden layers with 10 neurons each (per the numerical experiments).
  - **Activation:** hyperbolic tangent (tanh) used in those layers.
  - **Loss / optimization:** mean squared error minimized with Adam optimizer. fileciteturn1file11
- Training hyperparameters (examples from Table 2):
  - **Epochs:** 5,000 for many examples (some high‑dim example used 1,000).
  - **Batch sizes / learning rates / regularization:** vary per example (the paper lists per‑example values in Table 2). See the original Table 2 for exact per‑example settings. fileciteturn1file11

---

## 4. Why convolution helps (intuition)
- Traditional MF NN approaches often only use *collocated* LF data (same input locations as HF), leading to under‑utilization of abundant LF data.
- MDA‑CNN’s convolutional receptive fields allow the network to **connect a HF datum to many nearby (or all) LF data**, learning point‑to‑domain or point‑to‑all relationships rather than point‑to‑point only.
- This enables the network to exploit global LF trends and local LF structure to improve HF predictions, especially when HF samples are scarce. fileciteturn1file4

---

## 5. Extensions / options described
- **Multiple LF sources:** the input table can be extended with more columns to include additional LF models or extra LF features (e.g., derivatives) — see Fig. 4 variants. fileciteturn1file12
- **Gradient information:** explicitly including LF derivatives in the input table can be necessary when the HF model depends on LF derivatives. The authors show this improves cases like phase‑shifted oscillations. fileciteturn1file16
- **Higher dimensional inputs:** for multi‑dimensional inputs `y ∈ ℝ^n`, the left side of the table contains LF info and the right side indicates the LF datum corresponding to the HF sample; convolution still operates over LF samples. fileciteturn1file12
- **Probabilistic extensions:** authors note Bayesian CNNs or uncertainty quantification is a future direction. fileciteturn1file1

---

## 6. Practical recipe (minimal reference implementation idea)
1. Build LF dataset `(y_L_j, Q_L(y_L_j))` for j=1..NL and HF dataset `(y_H_i, Q_H(y_H_i))` for i=1..NH.
2. For each HF index `i` construct an input table with columns `[y_L, Q_L(y_L), y_H_i, Q_L(y_H_i)]` and NL rows.
3. Define a 1‑D convolutional layer with kernel height `k_rows` (e.g., 3 rows) that slides over the NL rows; use `F` filters (e.g., 64).
4. Flatten feature maps and feed into FC network with a skip connection for linear mapping and `L` nonlinear FC layers (e.g., 2 layers × 10 units, `tanh`).
5. Train end‑to‑end with Adam minimizing MSE. fileciteturn1file6turn1file11

---

## 7. Key takeaways
- MDA‑CNN repurposes convolutional ideas to let each HF target "see" many LF data points via sliding receptive fields.
- It outperforms single‑fidelity NN and multi‑fidelity NN that only use collocated LF data in the paper's numerical and engineering examples.
- The input‑table construction is central: design it carefully to include any LF-derived features (derivatives, multiple LF models) you believe help relate LF → HF. fileciteturn1file3turn1file12

---

### Source
Summary derived from the uploaded manuscript (Manuscriptdraftmultifidelity.pdf). fileciteturn1file0
