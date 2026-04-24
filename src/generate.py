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
def generate(start="h", max_new_tokens=100):
    x = torch.tensor([tokenizer.encode(start)])

    for _ in range(max_new_tokens):
        logits = model(x[:, -8:])
        probs = torch.softmax(logits[:, -1, :], dim=-1)

        next_token = torch.multinomial(probs, 1)
        x = torch.cat([x, next_token], dim=1)

    return tokenizer.decode(x[0].tolist())

print(generate("h"))