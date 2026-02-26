# train.py
import torch
from dataset import get_batch, stoi, itos
from model import MiniGPT
from config import learning_rate, max_iters, eval_interval, device, batch_size
from dataset import vocab_size
import os

model = MiniGPT().to(device)
print("Parameters:",
      sum(p.numel() for p in model.parameters()) / 1e6,
      "Million")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=max_iters
)
@torch.no_grad()
def estimate_loss():
    model.eval()
    losses = []
    for _ in range(50):
        xb, yb = get_batch('train')
        _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)

start_iter = 0
checkpoint_path = "checkpoints/checkpoint_93000.pt"
checkpoint = torch.load(checkpoint_path, map_location=device)

model.load_state_dict(checkpoint['model_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_iter = checkpoint['iter']

for iter in range(start_iter, max_iters):
    xb, yb = get_batch("train")
    xb, yb = xb.to(device), yb.to(device)

    logits, loss = model(xb, yb)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()

    # print loss
    if iter % eval_interval == 0:
        avg_loss = estimate_loss()
        print(f"Iteration {iter}, Avg Train Loss: {avg_loss:.4f}")

    # save checkpoint
    if iter % 1000 == 0 and iter > 0:
        os.makedirs("checkpoints", exist_ok=True)
        checkpoint_path = f"checkpoints/checkpoint_{iter}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "iter": iter
        }, checkpoint_path)
        print(f"Checkpoint saved at iteration {iter}")