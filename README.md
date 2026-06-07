# 🤖 Sentiment Analysis using Fine-tuned BERT

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A deep learning project that fine-tunes **BERT (bert-base-uncased)** on the IMDb movie reviews dataset to classify text sentiment as **Positive** or **Negative** with high accuracy using the Hugging Face Transformers library.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Demo](#demo)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Inference](#inference)
- [Key Hyperparameters](#key-hyperparameters)
- [Tech Stack](#tech-stack)
- [References](#references)

---

## 📌 Overview

This project demonstrates **transfer learning with BERT** for Natural Language Processing. Instead of training a model from scratch, we take Google's pre-trained BERT model (trained on BooksCorpus + English Wikipedia) and fine-tune it on a downstream sentiment classification task.

**What makes this different from basic sentiment analysis:**
- Uses bidirectional context (BERT reads text left-to-right AND right-to-left simultaneously)
- Leverages 110M pre-trained parameters — requires only 2–4 epochs to converge
- Achieves state-of-the-art accuracy with minimal training data

---

## 🎬 Demo

```
Input:  "The cinematography was breathtaking and the story kept me hooked till the very end."
Output: ✅ Positive (confidence: 97.3%)

Input:  "Terrible script, wooden acting, and a plot that made no sense whatsoever."
Output: ❌ Negative (confidence: 94.1%)
```

---

## 📁 Project Structure

```
bert-sentiment-analysis/
│
├── data/
│   └── README.md               # Dataset download instructions
│
├── src/
│   ├── dataset.py              # Custom Dataset class and tokenization
│   ├── model.py                # BertForSequenceClassification wrapper
│   ├── train.py                # Training loop with validation
│   ├── evaluate.py             # Metrics: F1, accuracy, confusion matrix
│   └── predict.py              # Inference on custom text
│
├── notebooks/
│   └── bert_sentiment.ipynb    # End-to-end Colab notebook
│
├── outputs/
│   ├── best_model/             # Saved model weights (best val F1)
│   ├── training_curves.png     # Loss and accuracy plots
│   └── confusion_matrix.png    # Evaluation confusion matrix
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

### 1. Clone the repository
### 2. Create a virtual environment (recommended)
### 3. Install dependencies

**requirements.txt:**
```
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
scikit-learn>=1.3.0
evaluate>=0.4.0
accelerate>=0.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
numpy>=1.24.0
```

### 4. Verify GPU availability

```python
import torch
print(torch.cuda.is_available())   # Should print: True
print(torch.cuda.get_device_name(0))
```

> **No GPU?** Use [Google Colab](https://colab.research.google.com) (free T4 GPU) or [Kaggle Notebooks](https://kaggle.com) (free P100 GPU).

---

## 📊 Dataset

This project uses the **IMDb Movie Reviews** dataset:

| Split | Samples | Classes |
|-------|---------|---------|
| Train | 25,000 | Positive / Negative |
| Test  | 25,000 | Positive / Negative |

The dataset is perfectly balanced (50/50 split per class) — no class weighting needed.

### Loading via Hugging Face Datasets
> Labels: `0 = Negative`, `1 = Positive`

---

## 🧠 Model Architecture

```
Input Text
    │
    ▼
BertTokenizerFast
(max_length=128, padding, truncation)
    │
    ▼
BERT (bert-base-uncased)
├── 12 Transformer Encoder Layers
├── 768-dimensional hidden states
└── [CLS] token output extracted
    │
    ▼
Dropout (p=0.1)
    │
    ▼
Linear Classifier Head  [768 → 2]
    │
    ▼
Softmax → [Negative, Positive]
```

**Key insight:** Only the `[CLS]` token's final hidden state is used for classification. It aggregates information from the entire input sequence through self-attention across all 12 layers.

---

## 🏋️ Training

### Run training

```bash
python src/train.py \
  --model_name bert-base-uncased \
  --dataset imdb \
  --epochs 3 \
  --batch_size 16 \
  --learning_rate 2e-5 \
  --max_length 128 \
  --output_dir outputs/best_model
```


### ⚠️ Critical training notes

| Parameter | Recommended Value | Why |
|-----------|------------------|-----|
| Learning rate | `2e-5` to `5e-5` | Higher LR destroys pre-trained weights |
| Epochs | `2–4` | BERT overfits quickly beyond this |
| Batch size | `16` or `32` | BERT is memory-heavy; use gradient accumulation if needed |
| Warmup | First 10% of steps | Prevents large early updates |
| Grad clipping | `max_norm=1.0` | Prevents exploding gradients |

---

## 📈 Evaluation

Run evaluation on the test set:

Metrics computed:
- **Accuracy** — overall correct predictions
- **F1 Score (macro)** — use this as the primary metric, not accuracy
- **Precision & Recall** — per class
- **Confusion Matrix** — visualized with seaborn

---

## 🏆 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | ~93.2% |
| F1 Score (macro) | ~93.1% |
| Precision | ~93.0% |
| Recall | ~93.2% |
| Training Time | ~25 min (Colab T4) |

> Results may vary slightly depending on random seed and hardware.

---

## 🔍 Inference

Run predictions on your own text:

```bash
python src/predict.py --text "This film was a masterpiece of modern cinema."
```

Or use the `predict()` function directly.

---

## 🔑 Key Hyperparameters

```python
CONFIG = {
    "model_name"    : "bert-base-uncased",
    "num_labels"    : 2,
    "max_length"    : 128,       # Truncate sequences longer than this
    "batch_size"    : 16,
    "learning_rate" : 2e-5,
    "weight_decay"  : 0.01,
    "epochs"        : 3,
    "warmup_ratio"  : 0.1,       # 10% of total steps for warmup
    "dropout"       : 0.1,
    "grad_clip"     : 1.0,
    "seed"          : 42
}
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| PyTorch 2.0+ | Deep learning framework |
| Hugging Face Transformers | BERT model and tokenizer |
| Hugging Face Datasets | IMDb dataset loading |
| scikit-learn | Evaluation metrics |
| Matplotlib / Seaborn | Visualization |
| Google Colab / Kaggle | Free GPU runtime |

---

## 📚 References

- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — Devlin et al., 2018
- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [IMDb Dataset on Hugging Face](https://huggingface.co/datasets/imdb)
- [Fine-Tuning BERT for Text Classification](https://arxiv.org/abs/1905.05583)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

> Feel free to fork, star ⭐, and build on top of this!
