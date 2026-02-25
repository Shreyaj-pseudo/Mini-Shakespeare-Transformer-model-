# MiniGPT Shakespeare

An educational exploration of building and training a GPT-style Transformer from scratch, focused on character-level language modeling to generate Shakespeare-style text.

---

## Overview

- **Objective:** Train a lightweight autoregressive Transformer model (MiniGPT) to generate text in the style of William Shakespeare.
- **Motivation:** Learn about Transformer architectures, autoregressive training, and text generation.
- **Scope:** Character-level text generation; not intended for production or factual content.

---

## Key Components

### 1. Model Architecture

| Property | Value |
|---|---|
| Type | Decoder-only Transformer (GPT-style) |
| Embedding | Character-level token + positional embeddings |
| Layers | 4 Transformer blocks |
| Attention Heads | 4 per block |
| Hidden Dimension | 256 |
| Output | Predicts next character in sequence |

The model implements multi-head self-attention, feed-forward layers, layer normalization, and dropout — fully from scratch using PyTorch.

---

### 2. Dataset

- **Source:** Complete works of Shakespeare (public domain)
- **Preprocessing:** Character-level tokenization, building `stoi` and `itos` mappings
- **Training/Validation Split:** Random sampling of sequences for batches

---

### 3. Training Pipeline

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Loss | Cross-entropy (next-character prediction) |
| Optimizer | AdamW |
| LR Scheduler | Cosine Annealing |
| Batch Size | 4 sequences |
| Context Length | 128 characters |
| Iterations | Up to 90,000 |
| Gradient Clipping | 1.0 |

Checkpoints are saved regularly to `checkpoints/`. Loss decreased from ~2.2 initially to ~1.7 at later stages, resulting in readable Shakespeare-style generation.

---

### 4. Generation

- **Prompt:** Any text string (e.g., `"ROMEO:"`)
- **Sampling:** Temperature-controlled softmax with optional top-k sampling
- **Max Tokens:** Configurable (default 300)
- **Output:** Character-level generated text in Shakespearean style

Example output:
```
Romeo: the some shall the come forther,
And a sire the make to hath a grach shall to what me
that with fall so so shall and and sir,
The be to he well the wo all the scand the dear night must the play.

DUKE OF GRERLARENCE:
Go, my leard, I so hen the here the death the fare your,
This my his soul though are the canto be love to the with compered
Of like sepore contle to my do my core
To the see to this be with not the cause stand some,
There to streeds he cold whom the so so to shore
```

---

### 5. Hugging Face Integration

The trained model is hosted publicly on Hugging Face: [Shreyaj MiniGPT Shakespeare](https://huggingface.co/Shreyaj-pseudo/shreyaj-mini-gpt-shakespeare)

```python
from huggingface_hub import hf_hub_download
import torch
from model import MiniGPT
from config import device
from dataset import stoi, itos

checkpoint_path = hf_hub_download(
    repo_id="Shreyaj-pseudo/shreyaj-mini-gpt-shakespeare",
    filename="final_model.pt",
    local_dir="./downloaded_model"
)

checkpoint = torch.load(checkpoint_path, map_location=device)
model = MiniGPT().to(device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

def generate(prompt, max_new_tokens=300):
    context = torch.tensor(
        [stoi[c] for c in prompt], dtype=torch.long
    ).unsqueeze(0).to(device)
    output = model.generate(context, max_new_tokens=max_new_tokens)
    return ''.join([itos[i] for i in output[0].tolist()])

print(generate("ROMEO:"))
```

---

### 6. Features & Limitations

**Features:**
- Lightweight GPT-style model for educational purposes
- Character-level Shakespearean text generation
- Easily extendable for more data or longer contexts
- Checkpoints downloadable from Hugging Face

**Limitations:**
- Trained on a single dataset (Shakespeare) → limited domain
- Small model → occasional incoherence and character-level errors
- Not suitable for production, modern NLP tasks, or factual reasoning

---

### 7. Tools & Technologies

- Python 3.x
- PyTorch
- Hugging Face Hub (`huggingface_hub`)
- NumPy
- Optional: GPU acceleration for faster training

---

### 8. Future Work

- Increase model size or number of layers for better generation
- Experiment with subword tokenization or BPE for improved coherence
- Add fine-tuning on modern English literature or other datasets
- Build a small web demo for interactive Shakespeare generation