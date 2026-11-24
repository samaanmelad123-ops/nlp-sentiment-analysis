# 🧠 NLP Sentiment Analysis — Arabic/English Text Classifier

A deep learning pipeline for **binary sentiment classification** (Positive / Negative) built from scratch using **GloVe word embeddings** and a **Feed-Forward Neural Network (FFNN)** in PyTorch.

---

## 🔍 Project Overview

The model converts text reviews into rich **308-dimensional vectors** by combining:
- **300D GloVe embeddings** (pretrained on Wikipedia + Gigaword)
- **8 hand-crafted sentiment features** (negation handling, polarity density, etc.)

It then trains a 4-layer FFNN with Batch Normalization and Dropout for regularization.

---

## 🏗️ Architecture

```
Input (308-D)
    ↓
Linear(308 → 256) + BatchNorm + ReLU + Dropout(0.4)
    ↓
Linear(256 → 128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Linear(128 → 64)  + BatchNorm + ReLU + Dropout(0.2)
    ↓
Linear(64 → 1) + Sigmoid
    ↓
Output: Positive / Negative
```

---

## ✨ Key Features

- 🔤 **GloVe (300D)** pretrained word embeddings via `gensim`
- 🔄 **Negation-aware** vectorization — "not bad" ≈ "good"
- 📐 **8 hand-crafted features**: review length, positive/negative density, negation density, negated-positive/negative counts, net polarity, sentiment density
- ⚡ **GPU support** (CUDA) with automatic CPU fallback
- 🎯 **Interactive prediction mode** — type any sentence and see results in real time

---

## 📁 Project Structure

```
nlp/
├── EMB.py          # GloVe embedding builder + feature extraction + train/test split
├── model.py        # FFNN architecture (SentimentNN)
├── train.py        # Model training loop
├── predict.py      # Inference + interactive demo
├── ds.py           # Dataset utilities
├── pp,py           # Preprocessing pipeline
├── X_train.npy     # Training embeddings
├── X_test.npy      # Test embeddings
├── y_train.npy     # Training labels
├── y_test.npy      # Test labels
└── README.md
```

---

## 🚀 Usage

### Step 1 — Generate Embeddings

```bash
python EMB.py
```

> This loads GloVe (300D) and builds 308-D feature vectors. Requires a preprocessed `cleand date.csv`.

### Step 2 — Train the Model

```bash
python train.py
```

### Step 3 — Run Predictions

```bash
python predict.py
```

**Example output:**
```
'this movie is amazing'
[████████████████░░░░] 83.4%
-> Positive

'this movie is not bad'
[███████████░░░░░░░░░] 61.2%
-> Positive

'I hate this film'
[░░░░░░░░░░░░░░░░░░░░] 7.1%
-> Negative
```

---

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

> **Note:** GloVe model (~900MB) is downloaded automatically on first run via `gensim.downloader`.

---

## 🛠️ Technologies Used

| Tool | Purpose |
|------|---------|
| `PyTorch` | Neural network implementation |
| `gensim` | GloVe pretrained word vectors |
| `NumPy` | Array operations & `.npy` data files |
| `scikit-learn` | Train/test split (stratified) |
| `tqdm` | Progress bars during embedding |

---

## 👤 Author

**Samaan Melad**  
GitHub: [@samaanmelad123-ops](https://github.com/samaanmelad123-ops)
