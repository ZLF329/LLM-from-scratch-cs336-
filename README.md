#  LLM language model Implementation

This repository contains my implementation of the core components of a Transformer LM, built entirely from scratch using PyTorch. The skeleton code including testing code is obtained from cs336 Stanford.

---

## ✅ Implemented Modules

- **Attention.py** — Multi-Head Self-Attention mechanism with Scaled Dot-Product Attention  
- **BPETokenizer.py** — Byte Pair Encoding tokenizer for text preprocessing  
- **Embedding.py** — Token and positional embedding layers  
- **Linear.py** — Custom linear transformation layer  
- **Loss.py** — Stable cross-entropy loss implementation using the log-sum-exp trick  
- **PositionwiseFeedForward.py** — Two-layer feedforward network with GELU activation  
- **RMSNorm.py** — Root Mean Square Layer Normalization  
- **Rope.py** — Rotary Positional Embedding (RoPE) utilities  
- **RotaryPositionalEmbedding.py** — Implementation of rotation-based positional encoding  
- **Transformer.py** — Full Transformer block combining attention, normalization, and feedforward layers  

---

## 
