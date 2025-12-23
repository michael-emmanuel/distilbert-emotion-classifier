# Multi-Class Emotion Classification with DistilBERT

### High-Performance NLP via Transformer Fine-Tuning

[Python 3.9+](https://www.python.org)
[Hugging Face](https://huggingface.co)
[PyTorch](https://pytorch.org)

## Project Overview

This project demonstrates a production-grade NLP pipeline for classifying text into six emotional states: **Sadness, Joy, Love, Anger, Fear, and Surprise**.

By leveraging **Transfer Learning** with a pre-trained **DistilBERT** architecture, this model achieves high accuracy while maintaining a smaller memory footprint and faster inference speeds than standard BERT models—making it ideal for real-time deployment.

---

## Key Technical Features

- **Transformer Architecture**: Fine-tuned `distilbert-base-uncased` using the Hugging Face `Trainer` API.
- **Efficient Tokenization**: Implemented WordPiece tokenization with dynamic padding and truncation.
- **Optimized Training**: Configured with Weight Decay (0.01) and AdamW optimization for robust convergence.
- **Model Interpretability**: Included a custom evaluation suite that generates a **Confusion Matrix** to identify semantic overlap between complex classes (e.g., 'Joy' vs. 'Love').

---

## Performance Metrics

The model is evaluated on the `dair-ai/emotion` dataset using:

- **Accuracy**: Overall correctness of the emotion predictions.
- **Weighted F1-Score**: Used to ensure performance is consistent even if class distributions are imbalanced.

| Metric      | Score |
| ----------- | ----- |
| Accuracy    | 91.5% |
| Weighted F1 | 0.915 |

---

## Tech Stack

Frameworks: PyTorch, Hugging Face Transformers
Data Handling: Hugging Face Datasets
Visualization: Seaborn, Matplotlib (for Confusion Matrix analysis)
Metrics: Scikit-Learn, Evaluate

---

## Repository Structure

```
├── src/
│ ├── train.py        # Fine-tuning logic & training loop
│ └── model_eval.py   # Performance visualization & error analysis
├── results/          # Model metrics and Confusion Matrix plots
└── requirements.txt  # Production dependencies
```

---

## Analysis & Insights

The included Confusion Matrix (results/confusion_matrix.png) provides deep insights into the model's behavior. By analyzing "false positives" between similar emotions like Anger and Fear, we can iterate on data augmentation strategies to further improve the model's linguistic nuance.

---

## Quick Start

```
Install: pip install -r requirements.txt
Train: python src/train.py
Evaluate: python src/model_eval.py
```
