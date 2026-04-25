import torch
import torch.nn as nn

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, block_size=64, d_model=64, num_layers=4):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, num_heads=4, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model)
                ),
                'ln1': nn.LayerNorm(d_model),
                'ln2': nn.LayerNorm(d_model),
            })
            for _ in range(num_layers)
        ])

        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape

        pos = torch.arange(T, device=x.device)
        x = self.token_embedding(x) + self.position_embedding(pos)

        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        for layer in self.layers:
            attn_out, _ = layer['attn'](x, x, x, attn_mask=causal_mask)
            x = layer['ln1'](x + attn_out)
            x = layer['ln2'](x + layer['ffn'](x))

        return self.head(x)