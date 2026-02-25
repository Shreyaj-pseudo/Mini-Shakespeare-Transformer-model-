# generate.py
import torch
from model import MiniGPT
from dataset import stoi, itos
from config import device
import sys
import glob
import os

# Model
model = MiniGPT().to(device)

# Checkpoint
checkpoint_files = sorted(glob.glob("../checkpoints/checkpoint_*.pt"), key=os.path.getmtime)
checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else checkpoint_files[-1]
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print(f"Loaded checkpoint: {checkpoint_path}")

# Start text
start_text = sys.argv[1] if len(sys.argv) > 1 else "Romeo"
context = torch.tensor([[stoi[c] for c in start_text]], dtype=torch.long).to(device)
model.eval()
# Generate
out = model.generate(context, max_new_tokens=300,
                     temperature=0.5,
                     top_k=20)[0].tolist()

# Print
generated_text = "".join([itos[i] for i in out])
print("\n=== Generated Text ===\n")
print(generated_text)
print("\n=====================\n")