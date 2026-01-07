# MichaelScott-GPT 👔📄

> "I usually start with a joke. I'm not gonna do that right now." - *Michael Scott*

A custom Transformer-based chatbot trained from scratch to mimic the unique personality of Michael Scott from *The Office*. This project implements the complete "Attention Is All You Need" architecture manually using PyTorch, without relying on pre-built model wrappers like HuggingFace's `AutoModel`.

## 🚀 Project Overview

This repository demonstrates the end-to-end process of building a Large Language Model (LLM) system:

1.  **Data Engineering**: Preparation of a dialogue dataset (`src` context -> `tgt` response).
2.  **Architecture**: Full implementation of the Transformer architecture (Encoder-Decoder).
3.  **Tokenization**: Integration of OpenAI's `tiktoken` (BPE) for efficient subword tokenization.
4.  **Training Strategy**:
    *   Local development and debugging on CPU/MPS (Mac).
    *   Scalable cloud training using **Modal** (serverless GPU infrastructure).
5.  **Inference**: A text generation engine with advanced sampling strategies (Temperature, Top-k, Top-p).

## 🧠 Model Architecture

The model is defined in `utils/transformer.py` and strictly follows the original Transformer design:

*   **Embeddings**: Learned input embeddings + Sinusoidal Positional Encodings.
*   **Multi-Head Attention**: Custom implementation of scaled dot-product attention with masking.
*   **Encoder-Decoder**:
    *   **Encoder**: Processes the input context.
    *   **Decoder**: Autoregressively generates the response, using cross-attention to attend to the encoder output.
*   **Normalization**: Layer Normalization and Residual Connections (Add & Norm) after each sub-layer.
*   **Feed Forward**: Position-wise Feed-Forward Networks.

### Configuration
We use a custom configuration (defined in `models/config_tiktoken_2.json`):
*   **d_model**: 512 (Embedding dimension)
*   **n_layers**: 6 (Encoder/Decoder layers)
*   **n_heads**: 8 (Attention heads)
*   **d_ff**: 2048 (Feed-forward dimension)
*   **Vocab Size**: ~100k (via `tiktoken`)

## 🛠️ Tools & Technologies

*   **Python 3.10+**
*   **PyTorch**: Core deep learning framework.
*   **Tiktoken**: Fast BPE tokenization (used by OpenAI models).
*   **Modal**: Serverless cloud platform for training on T4/A10G GPUs.
*   **Tqdm**: Progress tracking.

## 📂 Project Structure

```bash
.
├── data/                    # Dataset files
│   ├── src.txt              # Input prompts/context
│   └── tgt.txt              # Michael Scott's responses
├── gpt_tokenizers/          # Tokenizer implementations
│   ├── tiktoken.py          # Wrapper for OpenAI's tiktoken
│   └── byte_level.py        # Custom BPE implementation
├── utils/                   # Model core
│   ├── transformer.py       # The massive Transformer class & sub-layers
│   ├── masks.py             # Attention masking logic (causal & padding)
│   └── data_loader.py       # PyTorch Dataset/DataLoader
├── training/                # Local training scripts
│   ├── train.py             # Main training loop
│   └── train.ipynb          # Notebook for experiments
├── modal/                   # Cloud infrastructure scripts
│   ├── train_modal_integrated.py  # Modal app for remote GPU training
│   ├── download_model.py          # Script to fetch weights from Modal
│   └── inference_improved.py      # Advanced inference script
├── models/                  # Saved weights and configs
│   ├── config_tiktoken_2.json
│   └── weights_tiktoken_2.pt (ignored in git)
└── generation/              # Inference scripts
```

## ⚡ Usage

### 1. Prerequisites

Ensure you have the required dependencies:

```bash
pip install torch tiktoken modal tqdm numpy
```

### 2. Training (Cloud via Modal)

We use Modal to train on high-end GPUs without local hardware constraints.

1.  **Setup Modal**:
    ```bash
    pip install modal
    modal setup
    ```

2.  **Run Remote Training**:
    ```bash
    modal run modal/train_modal_integrated.py
    ```
    This will:
    *   Upload your code and data.
    *   Provision a container with a GPU (T4/A10G).
    *   Train the model.
    *   Save the weights to a persistent network volume.

3.  **Download Weights**:
    ```bash
    modal run modal/download_model.py
    ```

### 3. Inference (Chat with Michael)

To generate responses using the trained model:

```bash
python modal/inference_improved.py
```

*Note: The script uses `inference_improved.py` which includes temperature scaling and top-p sampling to avoid repetitive loops ("That's what she said" loops).*

## 📈 Training Progression

1.  **Initial Attempt**: Simple BPE tokenizer, small model size. Resulted in high loss.
2.  **Refinement**: Switched to `tiktoken` for better token coverage.
3.  **Scaling**: Moved from local CPU training to Modal GPUs, allowing for larger batch sizes and more epochs.
4.  **Tuning**: Adjusted learning rate and added dropout to prevent overfitting on the relatively small dataset.

## 📝 Credits

Built by **Idris** as a deep dive into LLM architecture.
Inspired by *The Office* (US).

