# ConViX-IR: Phase 2 — ConvNeXt-Tiny CNN Baseline

**Context-Verified Iterative Re-Inspection CNN–ViT Framework for Calibrated Peach Freshness Assessment**

---

## 1. Phase 2 Objective

Phase 2 establishes the foundational single-modality convolutional neural network (CNN) baseline for peach freshness assessment using a pretrained **ConvNeXt-Tiny** architecture. It provides:
1. Standard CNN representation benchmarks.
2. Direct feature extraction ($E_C \in \mathbb{R}^{768}$) for downstream comparisons.
3. Clean, calibrated baseline probabilities ($p_C$) and error metrics prior to multi-modal ViT integration.

> **Research Status & Scope Boundary:**  
> Phase 2 is an independent ConvNeXt-Tiny CNN baseline. It does not implement CNN–ViT fusion, context verification, disagreement measurement, dynamic fusion, or iterative re-inspection.

---

## 2. Dataset Formulation

The pipeline utilizes the existing, verified dataset splits:
$$\mathcal{D} = \mathcal{D}_{\text{train}} \cup \mathcal{D}_{\text{val}} \cup \mathcal{D}_{\text{test}}$$

- **Total Samples:** 593 images
  - $\mathcal{D}_{\text{train}}$: 415 images (`Fresh_Peach` = 175, `Rotten_Peach` = 240)
  - $\mathcal{D}_{\text{val}}$: 89 images (`Fresh_Peach` = 37, `Rotten_Peach` = 52)
  - $\mathcal{D}_{\text{test}}$: 89 images (`Fresh_Peach` = 38, `Rotten_Peach` = 51)
- **Class Labels:**
  - $\text{Fresh\_Peach} \to 0$
  - $\text{Rotten\_Peach} \to 1$ (Positive Class)

---

## 3. Input Representation

Each input sample is an RGB image tensor:
$$x \in \mathbb{R}^{3 \times 224 \times 224}$$

---

## 4. Preprocessing Pipelines

### Training Preprocessing ($\mathcal{D}_{\text{train}}$)
Data augmentation is applied only during training:
1. `RandomResizedCrop(224, scale=(0.8, 1.0))`
2. `RandomHorizontalFlip(p=0.5)`
3. `RandomRotation(degrees=15)`
4. `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)`
5. `ToTensor()`
6. `Normalize(mean, std)`

### Validation & Test Preprocessing ($\mathcal{D}_{\text{val}}, \mathcal{D}_{\text{test}}$)
Deterministic preprocessing without random augmentation:
1. `Resize(256)`
2. `CenterCrop(224)`
3. `ToTensor()`
4. `Normalize(mean, std)`

### ImageNet Normalization
Standard channel-wise statistics:
$$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
$$x_{\text{norm}}[c] = \frac{x[c] - \mu_c}{\sigma_c}$$

---

## 5. ConvNeXt-Tiny Feature Extraction

The ConvNeXt-Tiny backbone processes $x$ through hierarchical 7x7 depthwise separable convolutional stages, LayerNorm, and inverted bottleneck blocks to produce the final spatial feature map $F_C(x)$.

---

## 6. Global Feature Representation $E_C$

Global Average Pooling (GAP) aggregates spatial activations across height and width:
$$E_C = \text{GAP}(F_C(x))$$

---

## 7. 768-D Feature Vector

The pooled representation is flattened to form a continuous 768-dimensional feature embedding:
$$E_C \in \mathbb{R}^{768}$$

---

## 8. Linear Classifier

The classification head is a direct linear projection without intermediate non-linearities, MLPs, or attention mechanisms:
$$z_C = W_C E_C + b_C$$

where:
$$W_C \in \mathbb{R}^{2 \times 768}, \quad b_C \in \mathbb{R}^2$$

---

## 9. Logit Equation

$$z_C = [z_{\text{Fresh}}, z_{\text{Rotten}}]^T \in \mathbb{R}^2$$

---

## 10. Softmax Probability Equation

Class probabilities are derived via the standard softmax function:
$$p_C(k) = \frac{\exp(z_C(k))}{\sum_{j=0}^1 \exp(z_C(j))}$$
$$p_C = [p_{\text{Fresh}}, p_{\text{Rotten}}], \quad p_{\text{Fresh}} + p_{\text{Rotten}} = 1$$

---

## 11. Prediction Equation

The categorical freshness prediction is obtained via maximum a posteriori (MAP) assignment:
$$\hat{y} = \arg\max_{k \in \{0, 1\}} p_C(k)$$

---

## 12. Cross-Entropy Training Objective

The model minimizes the standard two-class categorical cross-entropy loss over $N$ training samples:
$$\mathcal{L}_{\text{train}}(\theta) = -\frac{1}{N} \sum_{i=1}^N \sum_{k=0}^1 y_{i,k} \log p_C(k; x_i, \theta)$$

No auxiliary or custom losses are introduced in Phase 2.

---

## 13. AdamW Optimization

Model parameters $\theta = \{\theta_{\text{backbone}}, W_C, b_C\}$ are updated using AdamW with decoupled weight decay:
$$\theta_{t+1} = \theta_t - \eta_t \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} - \eta_t \lambda \theta_t$$
- Default initial learning rate: $\eta = 10^{-4}$
- Weight decay: $\lambda = 10^{-2}$

---

## 14. Cosine Learning-Rate Schedule

Learning rate $\eta_t$ follows a half-wave cosine decay:
$$\eta_t = \eta_{\min} + \frac{1}{2} (\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$
over total epochs $T$, decaying to $\eta_{\min} = 10^{-6}$.

---

## 15. Validation Model-Selection Logic

Validation data $\mathcal{D}_{\text{val}}$ is strictly reserved for monitoring generalization and checkpoint selection:
$$\theta^* = \arg\min_\theta \mathcal{L}_{\text{val}}(\theta)$$

Whenever an epoch yields an improved validation loss, the checkpoint `best_convnext_tiny.pt` is updated. Test data is never used for checkpoint selection.

---

## 16. Test Evaluation Protocol

After training completes, the best validation-loss checkpoint $\theta^*$ is restored and evaluated once on the unseen test set $\mathcal{D}_{\text{test}}$. Test metrics are computed directly on test predictions.

---

## 17. Evaluation Metrics

1. **Accuracy:** $\frac{TP + TN}{TP + TN + FP + FN}$
2. **Precision:** $\frac{TP}{TP + FP}$
3. **Recall (Sensitivity):** $\frac{TP}{TP + FN}$
4. **F1-Score:** $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
5. **Specificity:** $\frac{TN}{TN + FP}$
6. **ROC-AUC:** Area under the ROC curve based on $P(\text{Rotten} \mid x) = p_C(1)$.
7. **Brier Score:** $\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_{\text{Rotten}, i} - y_i)^2$
8. **Expected Calibration Error (ECE):** Partitioning confidences into $M=10$ equal-width bins:
   $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} |\text{acc}(B_m) - \text{conf}(B_m)|$$
9. **Confusion Matrix:** $TN, FP, FN, TP$ layout.

---

## 18. Reproducibility

- Configurable random seed (`--seed 42`).
- Fixed seeds for `random`, `numpy`, `torch.manual_seed`, `torch.cuda.manual_seed_all`.
- Deterministic CuDNN backend enabled.

---

## 19. Output Files (in `phase2_results/`)

| File | Format | Contents |
| :--- | :--- | :--- |
| `best_convnext_tiny.pt` | PyTorch Binary | Best model weights, optimizer state, epoch, validation loss |
| `training_history.csv` | CSV | `epoch`, `train_loss`, `train_accuracy`, `val_loss`, `val_accuracy`, `learning_rate` |
| `test_predictions.csv` | CSV | `image_path`, `true_label`, `predicted_label`, `fresh_probability`, `rotten_probability`, `logit_fresh`, `logit_rotten` |
| `test_metrics.json` | JSON | `accuracy`, `precision`, `recall`, `f1`, `specificity`, `roc_auc`, `brier_score`, `ece`, `confusion_matrix` |
| `confusion_matrix.csv` | CSV | $2 \times 2$ confusion matrix table |
| `phase2_summary.txt` | Text | Executive summary of test performance and parameters |
| `features/` *(optional)* | NPY / JSON | 768-D feature arrays $E_C$ when `--save_features` is passed |

---

## 20. Limitations

1. **Single-Modality CNN Focus:** Relies purely on convolutional inductive biases without multi-scale patch attention.
2. **Context Agnostic:** Evaluates isolated peach images without surrounding environmental or physical session context.
3. **Static Inference:** Does not iteratively re-inspect ambiguous or borderline samples (reserved for later CIRF phases).
