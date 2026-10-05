# ConViX-IR

**Context-Verified Iterative Re-Inspection CNN–ViT Framework for Calibrated Peach Freshness Assessment**

---

## 📌 Project Overview

**ConViX-IR** is an independent deep-learning research project focusing on calibrated peach freshness assessment using vision models.

> **Research Status Disclaimer:**  
> The proposed CIRF/ConViX-IR research architecture is under development. The current repository initially contains the dataset audit and CNN baseline components. Later research components are not yet implemented and have not yet been experimentally validated.

---

## 🚦 Research Roadmap & Current Status

- [x] **Repository & Project Architecture Setup**
- [ ] **Phase 1 — Dataset verification and leakage audit**
- [ ] **Phase 2 — ConvNeXt-Tiny baseline**
- [ ] **Phase 3+ — Iterative Re-Inspection CNN–ViT Integration & Calibration (Planned)**

---

## 📁 Repository Structure

```
conVix-IR/
│
├── README.md             # Project documentation and research roadmap
├── .gitignore            # Excludes datasets, model weights, checkpoints, & secrets
├── requirements.txt      # Core Python dependencies (PyTorch, timm, etc.)
│
├── src/
│   ├── phase1/           # Phase 1: Dataset audit and leakage verification modules
│   └── phase2/           # Phase 2: ConvNeXt-Tiny baseline training and evaluation
│
├── notebooks/            # Jupyter/Colab experimentation notebooks
├── configs/              # Experiment and hyperparameter configurations
├── results/              # Local experiment evaluation outputs (git-ignored)
├── checkpoints/          # Local model checkpoint weights (git-ignored)
└── docs/                 # Extended research notes and technical documentation
```

---

## 🔒 Dataset Storage & Policy

The peach freshness dataset is managed externally in Google Drive:
- **Drive Path:** `/content/drive/MyDrive/conVix-IR/dataset`

> **Note:** The actual dataset images, annotations, and heavy model weights are strictly excluded from this Git repository to comply with version control best practices and storage policies.

---

## ⚙️ Installation & Environment Setup

```bash
# Clone the repository
git clone https://github.com/<your-username>/conVix-IR.git
cd conVix-IR

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 📜 License & Citation

Research codebase under active development.
