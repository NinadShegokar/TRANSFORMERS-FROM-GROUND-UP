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

We trained the model on the **rajpurkar/squad** dataset for question answering. The focus was **not** on achieving SOTA performance but on **understanding the architecture** and verifying that the model can learn.

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

## 📊 Results (Learning, Not Winning)

| Metric       | Training | Validation |
|--------------|----------|------------|
| Loss         | 4.21     | 4.85       |
| Accuracy     | 18%      | 12%        |

**Note:** These results show the model learns *something* - but don't expect miracles! The goal was validation, not performance. For better results you'd need:
- More layers and heads 🧠
- Longer training time ⏳  
- Proper QA evaluation metrics 📊  

---

## 🏗️ Code Structure

```
├── Transformer.ipynb          # Main notebook with implementation
├── README.md                  # This file
```

Key components:
- **`Embedding`**: Word embeddings + positional encoding
- **`MultiHeadAttention`**: Self/cross-attention mechanisms
- **`Encoder/Decoder`**: Stacks of attention + FFN layers
- **`ProjectionLayer`**: Final output layer

---

## 🏋️ Training

To train the QA model:
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

## 🤔 Why This Exists

1. **Learn Transformers** by building every component
2. **Verify learning** on a real QA task
3. **Create reference code** for others

---

## 🎯 Future Work

- Add span-based QA evaluation
- Implement attention masking
- Try different attention variants

---

## 📜 License

MIT - use freely but don't expect production-grade results!
