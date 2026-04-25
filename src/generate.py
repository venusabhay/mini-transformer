import os
import torch
from model import MiniTransformer
from tokenizer import CharTokenizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "data", "data.txt")) as f:
    text = f.read()
tokenizer = CharTokenizer(text)

model = MiniTransformer(tokenizer.vocab_size)
model.load_state_dict(torch.load(os.path.join(BASE_DIR, "outputs", "checkpoints", "model.pt"), weights_only=True))
model.eval()

@torch.no_grad()
def generate(start="h", max_new_tokens=100, temperature=1.0, top_k=None):
    x = torch.tensor([tokenizer.encode(start)])

    for _ in range(max_new_tokens):
        logits = model(x[:, -64:])
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            top_values, _ = torch.topk(logits, top_k)
            logits[logits < top_values[:, -1:]] = float('-inf')

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, 1)
        x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())

print(generate("h", temperature=0.8, top_k=10))