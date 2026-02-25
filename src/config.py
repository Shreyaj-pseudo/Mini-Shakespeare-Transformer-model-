import torch
block_size = 128      # context length
batch_size = 4
n_embd = 256          # embedding dimension
n_head = 4           # number of attention heads
n_layer = 4          # number of transformer blocks
learning_rate = 3e-4
max_iters = 95000
eval_interval = 500
device = "cuda" if torch.cuda.is_available() else "cpu"
vocab_size = 65