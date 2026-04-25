import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from model import MiniTransformer
from tokenizer import CharTokenizer
from utils import get_batch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# load data
with open(os.path.join(BASE_DIR, "data", "data.txt"), "r") as f:
    text = f.read()

tokenizer = CharTokenizer(text)
data = torch.tensor(tokenizer.encode(text))

# model
model = MiniTransformer(tokenizer.vocab_size)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

block_size = 64
losses = []

for step in range(5000):
    xb, yb = get_batch(data, block_size)

    logits = model(xb)

    loss = loss_fn(
        logits.view(-1, tokenizer.vocab_size),
        yb.view(-1)
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if step % 100 == 0:
        print("step:", step, "loss:", loss.item())

    if step % 1000 == 0:
        checkpoint_dir = os.path.join(BASE_DIR, "outputs", "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"model_{step}.pt"))

# plot loss
plt.plot(losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.savefig(os.path.join(BASE_DIR, "outputs", "loss.png"))
plt.close()

# save final model
checkpoint_dir = os.path.join(BASE_DIR, "outputs", "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model.pt"))