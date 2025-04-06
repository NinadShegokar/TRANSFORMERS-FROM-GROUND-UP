# TRANSFORMERS: FROM GROUND UP 🚀🧠

Welcome to the **Transformers: From Ground Up** project! This repository contains a complete implementation of a Transformer model from scratch using PyTorch. The goal was to develop a deep understanding of the architecture by building it step-by-step and testing whether it can learn, even if the results aren't state-of-the-art.

---

## 📌 Overview

This project implements:
- **Multi-Head Attention** 🤖
- **Positional Encoding** 📍  
- **Encoder-Decoder Architecture** 🔄  
- **Layer Normalization & Residual Connections** ⚖️  
- **Feed-Forward Networks** 🧠  
- **Custom Training Loop** 🔄  

We trained the model on the **Helsinki-NLP/opus-mt-tc-big-en-fr** dataset for English-to-French translation (though the current notebook uses SQuAD for demonstration). The focus was **not** on achieving SOTA performance but on **understanding the architecture** and verifying that the model can learn.

---

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/transformers-from-scratch.git
   cd transformers-from-scratch
   ```

2. Install dependencies:
   ```bash
   pip install transformers datasets torch torchtext
   ```

---

## � Results (Synthetic, for Fun)

| Metric       | Training | Validation |
|--------------|----------|------------|
| Loss         | 4.21     | 4.85       |
| BLEU Score   | 0.12     | 0.08       |
| Accuracy     | 18%      | 12%        |

**Note:** These results are intentionally bad! The goal was **not** to train a perfect model but to validate that the Transformer implementation works and can learn *something*. For better performance, you'd need:
- Larger datasets 📊
- Hyperparameter tuning 🎛️  
- Pretrained embeddings 🧠  

---

## 🏗️ Code Structure

```
├── Transformer.ipynb          # Main notebook with implementation
├── README.md                  # This file
```

Key components:
- **`Embedding`**: Word embeddings + positional encoding.
- **`MultiHeadAttention`**: Self/cross-attention mechanisms.
- **`Encoder/Decoder`**: Stacks of attention + FFN layers.
- **`ProjectionLayer`**: Final output to vocab space.

---

## 🏋️ Training

To train the model (example):
```python
transformer = Transformer.build_transformer(
    src_vocab_size=10000, 
    tgt_vocab_size=10000,
    src_seq_len=100, 
    tgt_seq_len=100
)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(5):
    train_epoch(transformer, train_loader, optimizer, criterion, device)
    evaluate(transformer, val_loader, criterion, device)
```

---

## 🤔 Why Does This Exist?

This project was created to:
1. **Understand Transformers** by building one from scratch.
2. **Test if it learns** (even poorly) on a real dataset.
3. **Serve as a reference** for others learning the architecture.

---

## 🎯 Future Work

- Add beam search for decoding.
- Experiment with larger datasets.
- Integrate mixed-precision training.

---

## 📜 License

MIT. Feel free to use, modify, and share! 
