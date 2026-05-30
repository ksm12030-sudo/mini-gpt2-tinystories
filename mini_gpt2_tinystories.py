!pip install torch transformers datasets tqdm matplotlib

import os
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from transformers import GPT2TokenizerFast
from datasets import load_dataset
from tqdm import tqdm

# 기본 설정

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# 결과 재현을 위한 seed 고정
torch.manual_seed(42)
random.seed(42)

# TinyStories 데이터 불러오기

NUM_SAMPLES = 20000

dataset = load_dataset("roneneldan/TinyStories")

# train split에서 앞의 20000개 story만 사용
texts = dataset["train"]["text"][:NUM_SAMPLES]

print("사용한 story 개수:", len(texts))
print("첫 번째 story 예시:")
print(texts[0][:500])

# 여러 story를 하나의 긴 corpus로 합침
corpus = "\n\n".join(texts)

print("전체 문자 수:", len(corpus))

# GPT-2 tokenizer

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token

# 긴 corpus tokenize 경고 방지
tokenizer.model_max_length = int(1e9)

# story 사이에 end-of-text token을 넣어줌
corpus = (tokenizer.eos_token + "\n\n").join(texts)

# 텍스트를 token id로 변환
ids = tokenizer.encode(corpus)

vocab_size = tokenizer.vocab_size
data = torch.tensor(ids, dtype=torch.long)

print("총 token 개수:", len(data))
print("vocab_size:", vocab_size)

class NextTokenDataset(Dataset):

    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


block_size = 128
batch_size = 16
dataset = NextTokenDataset(data, block_size)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

# batch shape 확인
xb, yb = next(iter(loader))

print("xb.shape:", xb.shape)
print("yb.shape:", yb.shape)

class Head(nn.Module):

    def __init__(self, emb_dim, head_size, block_size, dropout=0.1):
        super().__init__()

        self.key = nn.Linear(emb_dim, head_size, bias=False)
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)

        # causal mask
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)      # (B, T, head_size)
        q = self.query(x)    # (B, T, head_size)
        v = self.value(x)    # (B, T, head_size)

        # attention score
        
        wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)

        # 미래 token 가리기
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # attention 확률
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # value 가중합
        out = wei @ v        # (B, T, head_size)

        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()

        assert emb_dim % num_heads == 0

        head_size = emb_dim // num_heads

        self.heads = nn.ModuleList([
            Head(emb_dim, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])

    
        self.proj = nn.Linear(emb_dim, emb_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 각 head 결과를 마지막 차원에서 이어붙임
        out = torch.cat([h(x) for h in self.heads], dim=-1)

        out = self.proj(out)
        out = self.dropout(out)

        return out

class FeedForward(nn.Module):

    def __init__(self, emb_dim, dropout=0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()

        self.ln1 = nn.LayerNorm(emb_dim)
        self.sa = MultiHeadAttention(emb_dim, num_heads, block_size, dropout)

        self.ln2 = nn.LayerNorm(emb_dim)
        self.ffwd = FeedForward(emb_dim, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class MiniGPT2(nn.Module):


    def __init__(
        self,
        vocab_size,
        block_size,
        emb_dim=256,
        num_heads=8,
        num_layers=6,
        dropout=0.1
    ):
        super().__init__()

        self.block_size = block_size

        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(block_size, emb_dim)

        self.blocks = nn.Sequential(*[
            Block(emb_dim, num_heads, block_size, dropout)
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(emb_dim)

        self.lm_head = nn.Linear(emb_dim, vocab_size, bias=False)

        self.drop = nn.Dropout(dropout)


    def forward(self, x):
        B, T = x.shape

        assert T <= self.block_size

        # 위치 index
        pos = torch.arange(T, device=x.device)

        # token embedding
        tok = self.token_embedding(x)           # (B, T, emb_dim)

        # position embedding
        pos = self.position_embedding(pos)[None]  # (1, T, emb_dim)

        # token 정보 + 위치 정보
        h = tok + pos
        h = self.drop(h)

        # Transformer blocks
        h = self.blocks(h)

        # final layer norm
        h = self.ln_f(h)

        # 다음 token 예측 logits
        logits = self.lm_head(h)                # (B, T, vocab_size)

        return logits

emb_dim = 256
num_heads = 8
num_layers = 6
dropout = 0.1

model = MiniGPT2(
    vocab_size=vocab_size,
    block_size=block_size,
    emb_dim=emb_dim,
    num_heads=num_heads,
    num_layers=num_layers,
    dropout=dropout
).to(device)

logits = model(xb.to(device))

print("logits.shape:", logits.shape)

def sequence_cross_entropy(logits, targets):
  
    return F.cross_entropy(logits.transpose(1, 2), targets)

def train_one_epoch(model, loader, optimizer, device, max_steps=None):
    model.train()

    total_loss = 0.0
    total_count = 0

    for step, (xb, yb) in enumerate(tqdm(loader)):
        xb = xb.to(device)
        yb = yb.to(device)

        logits = model(xb)
        loss = sequence_cross_entropy(logits, yb)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        total_count += xb.size(0)

        if max_steps is not None and step + 1 >= max_steps:
            break

    return total_loss / total_count

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(20):
    train_loss = train_one_epoch(
        model,
        loader,
        optimizer,
        device,
        max_steps=300
    )

    print(f"epoch {epoch:2d} | train loss {train_loss:.4f}")

def top_k_filter(logits, top_k):

    if top_k is None:
        return logits

    values, _ = torch.topk(logits, top_k)
    min_values = values[:, -1].unsqueeze(1)

    logits = torch.where(
        logits < min_values,
        torch.full_like(logits, float("-inf")),
        logits
    )

    return logits

@torch.no_grad()
def sample_gpt(model, block_size, tokenizer, device,
               start_text="Once upon a time",
               max_new_tokens=400,
               temperature=0.8,
               top_k=50):

    model.eval()

    input_ids = tokenizer.encode(start_text)

    context = torch.zeros((1, block_size), dtype=torch.long, device=device)

    for token_id in input_ids:
        ix = torch.tensor([[token_id]], dtype=torch.long, device=device)
        context = torch.cat([context[:, 1:], ix], dim=1)

    out_ids = input_ids.copy()

    for _ in range(max_new_tokens):
        logits = model(context)
        logits = logits[:, -1, :]

        # temperature 적용
        logits = logits / temperature

        # top-k 적용
        logits = top_k_filter(logits, top_k)

        probs = F.softmax(logits, dim=-1)
        ix = torch.multinomial(probs, num_samples=1)

        out_ids.append(ix.item())

        context = torch.cat([context[:, 1:], ix], dim=1)

    return tokenizer.decode(out_ids)

print(
    sample_gpt(
        model,
        block_size,
        tokenizer,
        device,
        start_text="Once upon a time",
        max_new_tokens=500,
        temperature=0.8,
        top_k=50
    )
)

