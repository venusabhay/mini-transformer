import torch
import torch.nn as nn

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, block_size=8, d_model=64):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        self.attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape

        pos = torch.arange(T, device=x.device)
        x = self.token_embedding(x) + self.position_embedding(pos)

        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        attn_out, _ = self.attn(x, x, x, attn_mask=causal_mask)
        x = self.ln1(x + attn_out)

        x = self.ln2(x + self.ffn(x))

        return self.head(x)