# Generating Mini GPT 
Mini GPT-2 style language model trained on OpenWebText using PyTorch

# 1. Project Overview

이 프로젝트는 GPT 계열 언어모델의 핵심 구조를 PyTorch로 직접 구현하는 것을 목표로 한다.

우리는 언어모델이 다음과 같이 발전하는 과정을 학습하였다.

```
Bigram Model
→ MLP on Names
→ MLP on Shakespeare
→ GPT-style Dataset + Mini Sequence Model
→ Single-head Masked Self-Attention
→ Multi-head Masked Self-Attention
→ Mini GPT-style Transformer Language Model
```

본 프로젝트에서는 이 흐름을 바탕으로 OpenWebText 일부 데이터를 사용하여 Mini GPT-style language model을 구현하였다.

최종 모델은 다음과 같은 구조를 가진다.

```
input token ids
→ token embedding
→ position embedding
→ Transformer blocks
   → LayerNorm
   → Multi-Head Masked Self-Attention
   → Residual Connection
   → LayerNorm
   → FeedForward Network
   → Residual Connection
→ final LayerNorm
→ lm_head
→ logits over vocabulary
→ next-token prediction
```

이 프로젝트의 핵심 목적은 단순히 텍스트를 생성하는 것이 아니라, Bigram부터 GPT-style Transformer까지 언어모델이 왜 이런 구조로 발전했는지, 그리고 각 단계에서 tensor dimension이 어떻게 변하는지 이해하는 것이다.

## 1.1 Main Variables

| 변수 | 의미 | 본 프로젝트 설정 |
| --- | --- | --- |
| `vocab_size` | tokenizer가 표현할 수 있는 token 종류 수 | 50,257 |
| `block_size` | 모델이 한 번에 볼 수 있는 최대 문맥 길이 | 256 |
| `batch_size` | 한 번에 학습하는 sequence 묶음 수 | 8 |
| `emb_dim` | token 하나를 표현하는 embedding vector 차원 | 256 |
| `num_heads` | multi-head attention에서 사용하는 head 개수 | 8 |
| `head_size` | attention head 하나가 사용하는 차원 | 32 |
| `num_layers` | Transformer block 개수 | 6 |
| `dropout` | 과적합 방지를 위해 일부 값을 랜덤하게 제거하는 비율 | 0.1 |
| `learning_rate` | optimizer가 parameter를 업데이트하는 크기 | 3e-4 |
| `max_tokens` | OpenWebText에서 수집한 token 수 | 5,000,000 |
| `max_steps` | 각 epoch에서 학습하는 최대 mini-batch 수 | 300 |

## 1.2 Requirements and Libraries

본 프로젝트에서 사용한 주요 라이브러리는 다음과 같다.

| 라이브러리 | 역할 |
| --- | --- |
| `torch` | PyTorch 기반 tensor 연산, neural network 구현, 학습 수행 |
| `torch.nn` | `nn.Module`, `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, `nn.Dropout` 등 모델 구성 요소 제공 |
| `torch.nn.functional` | `softmax`, `cross_entropy` 등 함수형 API 제공 |
| `torch.utils.data.Dataset` | next-token prediction dataset을 직접 정의하기 위해 사용 |
| `torch.utils.data.DataLoader` | batch 단위로 데이터를 불러오기 위해 사용 |
| `transformers` | GPT-2 tokenizer를 불러오기 위해 사용 |
| `datasets` | Hugging Face의 OpenWebText 데이터를 streaming 방식으로 불러오기 위해 사용 |
| `tqdm` | OpenWebText token 수집 과정의 진행률 표시 |
| `matplotlib` | training loss curve 시각화 |
| `math` | 수학 연산에 사용 |
| `os`, `pathlib` | 파일 경로 및 저장 여부 확인 |

설치 예시는 다음과 같다.

```
pip install torch transformers datasets tqdm matplotlib
```
또는 `requirements.txt`를 사용하는 경우:

```
pip install-r requirements.txt
```

---

# 2. Development Process of Language Models

## 2.1 Bigram Model: `[1] → [1]`

Bigram model은 가장 단순한 언어모델이다. 현재 token 하나만 보고 다음 token을 예측한다.

예를 들어 :

```
"finance engineering" 라는 문자열에서 

"f" -> "i"
"i" -> "n"
```

와 같은 관계를 학습한다.

dimension 관점에서는 입력과 출력이 다음과 같다.

```
input:  (B, T) / T = 1 
W.shape = [vocab_size, vocab_size]
output: (B, T, vocab_size)
```

### 한계

Bigram model의 가장 큰 한계는 **이전 문맥을 거의 보지 못한다는 것**이다. 현재 token 하나만을 보고 다음 token을 예측하므로, 긴 문장 구조, 의미적 관계, 문맥적 흐름을 학습하기 어렵다.

---

## 2.2 MLP: [n] → [1]에서 [n] → [n]으로

Bigram model은 바로 앞 token 하나만 보고 다음 token을 예측한다는 한계가 있다. 이를 보완하기 위해 여러 개의 이전 token을 context로 사용하는 MLP model로 확장할 수 있다.

MLP는 Multi-Layer Perceptron의 약자로, 여러 개의 층으로 구성된 기본적인 신경망 모델이다. 언어 모델에서는 여러 token을 embedding으로 바꾼 뒤, 이 정보를 이용해 다음 token을 예측한다.

### MLP on Names: [n] → [1]

Names dataset에서는 여러 개의 이전 글자를 보고 다음 글자 하나를 예측한다.

예를 들어 `anna`라는 이름을 학습한다고 하면 다음과 같다.

context = [a, n, n]
target  = a

즉, 여러 글자를 입력으로 받아 다음 글자 하나를 예측하는 구조이다.

dimension 관점에서 보면:

```
batch_size = 32
block_size = 3
emb_dim = 10
```

일 때 입력은 다음과 같다.

```
x.shape = (32, 3)
```

embedding 후에는 각 token이 emb_dim 크기의 벡터로 바뀐다.

```
embedding.shape = (32, 3, 10)
```

MLP에 넣기 위해 context 전체를 하나의 벡터로 펼치면 다음과 같다.

```
(32, 3, 10) → (32, 30)
```

최종 출력은 다음 token 후보들에 대한 점수이다.

```
logits.shape = (32, vocab_size)
```

따라서 MLP on Names는 다음과 같은 구조로 이해할 수 있다.

```
여러 개의 이전 token → embedding → flatten → MLP → 다음 token 예측
```

### MLP on Shakespeare: `[n] → [n]`

Names dataset은 짧은 이름을 생성하는 task였지만, Shakespeare dataset은 긴 문장, 대화, 문체, 문맥이 포함된 더 복잡한 text dataset이다. 따라서 Shakespeare에서는 여러 token을 입력으로 받아, 여러 위치에서 각각 다음 token을 예측하는 방식으로 확장할 수 있다.

예를 들어:

```
x = [t1, t2, t3, t4]
y = [t2, t3, t4, t5]
```

dimension 관점에서는 다음과 같다.

```
x.shape = (B, T)
y.shape = (B, T)
```

embedding 후에는 다음과 같은 형태가 된다.

```
embedding.shape = (B, T, emb)
```

## MLP의 한계

MLP는 Bigram보다 더 긴 context를 볼 수 있다는 장점이 있지만, sequence를 처리하는 방식에는 한계가 있다.

가장 큰 한계는 context를 하나의 긴 벡터로 펼쳐서 사용한다는 점이다.

```
(B, T, emb) → (B, T*emb)
```

이 방식은 `block_size`가 커질수록 입력 차원이 빠르게 증가한다. 예를 들어 `T=3, C=10`이면 flatten 후 30차원이지만, `T=100, C=256`이면 25,600차원이 된다. 따라서 긴 문맥을 다루려 할수록 계산량과 parameter 수가 크게 증가한다.

또한 MLP는 각 token이 현재 문맥에서 서로 얼마나 관련 있는지 직접 비교하지 못한다. 고정된 위치의 token들을 한꺼번에 펼쳐서 사용하기 때문에, 문장마다 어떤 이전 token이 더 중요한지 유연하게 판단하기 어렵다.

---

## 2.3 GPT-style Dataset + Token Embedding + Position Embedding

GPT-style language model은 sequence 전체에 대해 next-token prediction을 수행한다.

```
input:  [t1, t2, t3, t4]
target: [t2, t3, t4, t5]
```

즉, 각 위치에서 다음 token을 예측한다.

```
t1             → t2
t1, t2         → t3
t1, t2, t3     → t4
t1, t2, t3, t4 → t5
```

dimension 관점에서 입력과 target은 다음과 같다.

```
x.shape = (B, T)
y.shape = (B, T)
```

### Token Embedding

입력 x의 차원은 다음과 같다. 

```
x.shape = (B, T)
```

예를 들어 각 값은 실제 단어가 아니라 token 번호이다.

```
x = [15, 23, 8, 91] (block_size = 4)
```

하지만 token 숫자 자체에는 의미가 없다. 따라서 모델이 학습할 수 있도록 각 token 숫자를 벡터로 바꿔줘야 한다. 이 역할을 하는 것이 token embedding이다.

```
self.token_embedding_table = nn.Embedding(vocab_size,emb_dim)
```

dimension은 다음과 같다.

```
token_embedding_table.shape = (vocab_size, emb_dim)
```

입력 `x`를 token embedding에 넣으면 :

```
tok_emb=self.token_embedding_table(idx)
```

dimension은 다음과 같이 바뀐다.

```
idx.shape     = (B, T)
tok_emb.shape = (B, T, C)
```

즉, token embedding은 각 token에게 “무슨 token인지”에 대한 정보를 부여한다.

### Position Embedding

Transformer는 token의 순서를 자동으로 알지 못한다. 따라서 각 token이 몇 번째 위치에 있는지 알려주는 position embedding이 필요하다.

```
self.position_embedding_table = nn.Embedding(block_size,emb_dim)
```

dimension은 다음과 같다.

```
position_embedding_table.shape = (block_size, emb_dim)
```

예를 들어 `block_size=128`, `emb_dim=256`이면:

```
position_embedding_table.shape = (128, 256)
```

즉, 0번째 위치부터 127번째 위치까지 각각 256차원 위치 벡터를 갖는다는 뜻이다.

즉, position embedding은 각 token에게 “몇 번째 위치인지”에 대한 정보를 부여한다.

### Mini Sequence Model과 Attention의 필요성

GPT-style dataset은 sequence 전체를 입력으로 사용하지만, 단순한 mini sequence model만으로는 token 간 관계를 충분히 학습하기 어렵다. 언어에서는 어떤 token이 중요한지 문맥마다 달라진다.

예를 들어 대명사가 어떤 명사를 가리키는지, 주어와 동사가 어떻게 연결되는지 등은 고정된 위치만으로 판단하기 어렵다.

따라서 각 token이 다른 token들과의 관계를 직접 계산할 수 있는 self-attention이 필요하다.

---

## 2.4 Single-head Masked Self-Attention

Self-attention은 sequence 안의 각 token이 다른 token들을 얼마나 참고할지 계산하는 구조이다.

Self - attention model은 각 token embedding으로부터 query, key, value를 만든다.

| 요소 | 역할 | dimension |
| --- | --- | --- |
| **Query (Q)** | 현재 token이 어떤 정보를 찾고 싶은지 나타냄 | `(B, T, head_size)` |
| **Key (K)** | 각 token이 어떤 정보를 가지고 있는지 나타냄 | `(B, T, head_size)` |
| **Value (V)** | 실제로 가져올 token 정보 | `(B, T, head_size)` |

각 token이 어떤 token을 얼마나 참고해야 할지를 나타내는 Attention score는 query와 key의 내적으로 계산된다.

```
wei=q@k.transpose(-2,-1)
```

dimension은 다음과 같다.

```
q.shape                  = (B, T, H)
k.transpose(-2, -1).shape = (B, H, T)
wei.shape                = (B, T, T)
```

여기서 `wei[b, i, j]`는 b번째 batch에서 i번째 token이 j번째 token을 얼마나 참고할지를 의미한다.

### Masked Attention

GPT는 다음 token을 예측하는 모델이므로 미래 token을 보면 안 된다.

따라서 lower triangular mask (하삼각행렬)을 사용한다.

```
torch.tril(torch.ones(block_size,block_size))
```

예를 들어 `block_size=4`라면 mask는 다음과 같다.

```
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

이 mask를 이용해 미래 위치를 `-inf`로 바꾸면, softmax 이후 해당 위치의 확률은 0이 된다.

```
wei=wei.masked_fill(self.tril[:T, :T]==0,float("-inf"))
```

그 후 softmax를 적용해 attention weight를 만들고 value와 곱한다.

```
wei=F.softmax(wei,dim=-1)
out=wei@v
```

최종 출력은 다음과 같다.

```
out.shape = (B, T, head_size)
```

---

## 2.5 Multi-head Masked Self-Attention

Single-head attention은 하나의 관점에서만 token 관계를 본다. 하지만 언어에서는 여러 관계가 동시에 중요하다.

Multi-head attention은 여러 개의 attention head를 병렬로 사용한다.

본 프로젝트에서는:

```
emb_dim=256
num_heads=8
head_size=32
```

이다.

각 head는 다음과 같은 출력을 만든다.

```
head output.shape = (B, T, 32)
```

8개의 head를 마지막 차원에서 이어붙이면:

```
(B, T, 32 * 8) = (B, T, 256)
```

그 후 projection layer를 통해 head들의 정보를 다시 섞는다.

---

## 2.6 Transformer Block
최종 GPT-style Transformer block은 다음 구조를 가진다.

```
x
→ LayerNorm
→ Multi-Head Masked Self-Attention
→ Residual Connection
→ LayerNorm
→ FeedForward Network
→ Residual Connection
```

| 개념 | 역할 |
| --- | --- |
| LayerNorm | 값의 분포를 안정화 |
| Residual connection | 입력을 출력에 더함 |

---

# 3. Implementation Details

## 3.1 Data Loading : OpenWebText 5M Tokens

본 프로젝트에서는 OpenWebText 일부 데이터를 사용하였다.

OpenWebText는 웹문서 기반 영어 text corpus이며, GPT-2의 WebText 학습 방식과 유사한 실험을 하기 위해 사용하였다.

전체 OpenWebText를 모두 학습하는 것은 Colab 환경에서 현실적으로 어렵기 때문에, 본 프로젝트에서는 streaming 방식으로 데이터를 순차적으로 읽고 GPT-2 tokenizer로 5,000,000개의 token만 수집하였다.

```
dataset=load_dataset(
"Skylion007/openwebtext",
split="train",
streaming=True
)
```

token 수집 과정은 다음과 같다.

```
max_tokens=5_000_000
all_ids= []

forexampleindataset:
text=example["text"]
ids=tokenizer.encode(text+tokenizer.eos_token)
all_ids.extend(ids)

iflen(all_ids)>=max_tokens:
break
```

각 문서 끝에는 GPT-2의 end-of-text token을 추가하였다.

```
text+tokenizer.eos_token
```

이를 통해 문서와 문서 사이의 경계를 모델이 학습할 수 있도록 하였다.

---

## 3.2 GPT-2 Tokenizer

본 프로젝트에서는 GPT-2 tokenizer를 사용하였다.

```
tokenizer=GPT2TokenizerFast.from_pretrained("gpt2")
```

GPT-2 tokenizer의 vocabulary size는 다음과 같다.

```
vocab_size=50257
```

이는 모델이 각 위치에서 예측해야 하는 token 후보가 50,257개라는 뜻이다.

입력 text는 tokenizer를 통해 token id sequence로 변환된다.

```
"In recent years,"
→ [token_id_1, token_id_2, token_id_3, ...]
```

---

## 3.3 Dataset and DataLoader

next-token prediction을 위해 `NextTokenDataset`을 직접 정의하였다.

```
classNextTokenDataset(Dataset):

def__init__(self,data,block_size):
self.data=data
self.block_size=block_size

def__len__(self):
returnlen(self.data)-self.block_size

def__getitem__(self,idx):
x=self.data[idx :idx+self.block_size]
y=self.data[idx+1 :idx+self.block_size+1]
returnx,y
```

DataLoader는 batch를 구성한다.

```
loader=DataLoader(
dataset,
batch_size=batch_size,
shuffle=True,
drop_last=True
)
```

본 프로젝트에서는 다음과 같이 설정하였다.

```
block_size=256
batch_size=8
```

따라서 batch shape은 다음과 같다.

```
xb.shape = (8, 256)
yb.shape = (8, 256)
```

---

## 3.4 Single Attention Head

`Head` class는 single masked self-attention head를 구현한다.

```
class Head(nn.Module):

    def __init__(self, emb_dim, head_size, block_size, dropout=0.1): #emb_dim, head_size(emb_dim을 몇개로 쪼갤건지), block size 변수로 저장
        super().__init__()

        self.key = nn.Linear(emb_dim, head_size, bias=False) # emb_dim -> head_size 차원 변경
        self.query = nn.Linear(emb_dim, head_size, bias=False)
        self.value = nn.Linear(emb_dim, head_size, bias=False)

        # causal mask
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size)) # trill이라는 이름으로 [T,T] 하삼각행렬 구성
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x): # 실제 계산
        B, T, C = x.shape # (C = emb_dim)

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

        return out  # (B,T,emb_dim) -> (B,T,head_size)
```

---

## 3.5 MultiHeadAttention

`MultiHeadAttention` class는 여러 개의 `Head`를 병렬로 실행한다.

```
class MultiHeadAttention(nn.Module):

    def __init__(self, emb_dim, num_heads, block_size, dropout=0.1):
        super().__init__()

        assert emb_dim % num_heads == 0   # emb_dim은 num_heads 로 나누어 떨어져야 한다.

        head_size = emb_dim // num_heads

        self.heads = nn.ModuleList([
            Head(emb_dim, head_size, block_size, dropout)   # num_heads 만큼 Head 만들기
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(emb_dim, emb_dim) #

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 각 head 결과를 마지막 차원에서 이어붙임
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # h(x) = (B, T, head_size) / cat -> (B,T,head_size * num_heads)

        out = self.proj(out)
        out = self.dropout(out)

        return out # (B,T,head_size) -> (B,T,emb_size)
```

`nn.ModuleList`를 사용하는 이유는 PyTorch가 여러 head의 parameter를 모델의 학습 대상 parameter로 인식하게 하기 위해서이다.

---

## 3.6 FeedForward

FeedForward는 학습이 너무 선형적으로만 이루어지지 않게 하기 위한 비선형 학습 모델이다. 

```
self.net=nn.Sequential(
nn.Linear(emb_dim,4*emb_dim),
nn.GELU(),
nn.Linear(4*emb_dim,emb_dim),
nn.Dropout(dropout),
)
```

dimension은 다음과 같다.

```
(B, T, 256)
→ Linear: (B, T, 1024)
→ GELU:   (B, T, 1024)
→ Linear: (B, T, 256)
→ Dropout:(B, T, 256)
```

---

## 3.7 Mini - GPT Language Model

최종 Mini - GPT Language Model은 다음 구성요소를 포함한다.

```
token embedding
position embedding
Transformer blocks
final LayerNorm
lm_head
```

최종 출력은 vocab 전체에 대한 logits이다.

```
logits.shape = (B, T, vocab_size)
```

본 프로젝트에서는 다음과 같다 :

```
logits.shape = (8, 256, 50257)
```

---

## 3.8 Training Loop

학습 과정은 다음과 같다.

```
1. batch에서 xb, yb를 가져온다.
2. xb를 모델에 넣어 logits를 계산한다.
3. logits와 yb를 비교해 cross entropy loss를 계산한다.
4. loss.backward()로 gradient를 계산한다.
5. optimizer.step()으로 parameter를 업데이트한다.
```

본 프로젝트에서는 각 epoch마다 전체 데이터셋을 모두 순회하지 않고, 계산 시간을 줄이기 위해 최대 300 mini-batch만 학습하였다.

```
max_steps=300
```

DataLoader의 `shuffle=True` 설정으로 인해 각 epoch마다 데이터 순서가 무작위로 섞인다. 따라서 매 epoch마다 서로 다른 batch 조합을 학습하게 된다.

---

## 3.9 Sampling Function

학습된 모델을 이용해 text generation을 수행하였다.

생성 과정은 다음과 같다.

```
1. start_text를 GPT-2 tokenizer로 token id로 변환한다.
2. block_size 길이의 context를 만든다.
3. 모델이 다음 token logits를 예측한다.
4. 마지막 위치의 logits만 사용한다.
5. temperature와 top-k를 적용한다.
6. softmax로 확률분포를 만든다.
7. torch.multinomial로 token 하나를 샘플링한다.
8. 생성된 token을 context 뒤에 붙인다.
9. 이 과정을 반복한다.
10. token id를 다시 text로 decode한다.
```

### Temperature

`temperature`는 생성의 다양성을 조절한다.

```
logits = logits / temperature
```

```
temperature 낮음 → 안정적, 반복적
temperature 높음 → 다양함, 불안정할 수 있음
```

본 프로젝트에서는 `temperature=0.8`을 사용하였다.

### Top-k Sampling

`top_k`는 점수가 높은 상위 k개 token만 후보로 남기는 방식이다.

```
top_k=50
```

즉, vocab 전체 50,257개 중에서 가장 가능성이 높은 50개 token 중 하나를 샘플링한다.

이를 통해 너무 낮은 확률의 이상한 token이 생성되는 것을 줄일 수 있다.

---

# 4. Results

## 4.1 Training Loss

OpenWebText 5M tokens를 사용해 학습한 결과, training loss는 약 7.55에서 시작해 100 epoch 후 약 4.33까지 감소하였다.

<img width="691" height="470" alt="training_loss" src="https://github.com/user-attachments/assets/c0c0fc3d-5663-4666-9012-7af51d89463e" />


이는 해당 모델이 loss를 줄여나가면서 학습을 했음을 의미한다. 

---

## 4.2 Sampling Result

학습된 모델로 다음과 같은 start text를 사용해 text generation을 수행하였다.

```
In recent years,
```

sampling 설정은 다음과 같다.

```
temperature=0.8
top_k=50
max_new_tokens=500
```

생성 결과 예시는 다음과 같다.

```
In recent years, I’ll be laughing!

Oh!

So!

So wait I’ve got out there to go up for the next couple. A little pictures

What’ The most thing I’ve done this film out for the next step we’ve been working for.

So I’ve got into the fun since I had a bunch of years old. I love it. We’ve been a lot of other places. I feel that we’ve done, but it’s quite funny.

If you’ve read the videos, then go ahead of the show. The story is a new movie that I have been so many fun to see if it’s gone.

I’ve always tried to put together it together, as I get it. You can’t let alone feel you can’t get you out there.

Yeah don’t forget anything like there for it.

This is my first, and I’ve enjoyed it from quite like I’m really excited to be out of it. I want to try it together and just get a bit fun to help you do something.

I always love the time. That’s because I’m really good. I’m getting excited that I’m going to say it’s not going to be a real life. It’s a good thing.

I was a very high quality, and I didn’t have a lot of the real life. I like to try a film, but I’ve just got to look at it.

I’m not a good thing, but I think that’s happening.

And it’s not a real shame. I mean, because it’s like me, I’m not going to be on it!

I really have to feel like something else I’ve been working on the whole life of my life. I never really liked the whole thing I’m going to look, but I’m not sure what the best I’ve ever been, I have been looking forward to talk to my life since I never had been. Then I’m really sure what I get past. I’m like to go. I think it

```

생성 결과는 완전한 GPT-2 수준으로 자연스럽지는 않지만, OpenWebText 기반의 일반 영어 문장 패턴을 일부 학습한 것을 확인할 수 있다.

문법적 불안정성이나 반복 표현은 모델 규모와 학습 데이터 규모가 제한적이기 때문에 발생한다.

---

# 5. Comparison with GPT-2 Small

본 프로젝트는 GPT-2의 구조를 참고해 구현한 Mini GPT language model이다.

하지만 실제 GPT-2 small과 비교하면 모델 규모와 데이터 규모가 훨씬 작다.

| 항목 | 본 프로젝트 | GPT-2 small |
| --- | --- | --- |
| tokenizer | GPT-2 tokenizer | GPT-2 tokenizer |
| vocab size | 50,257 | 50,257 |
| training data | OpenWebText 일부 | WebText |
| token/data scale | 5M tokens | 훨씬 큰 대규모 corpus |
| block size / context length | 256 | 1024 |
| batch size | 8 | 512 |
| embedding dimension | 256 | 768 |
| number of heads | 8 | 12 |
| head size | 32 | 64 |
| number of layers | 6 | 12 |
| objective | next-token prediction | next-token prediction |
| pretrained weights | 사용하지 않음 | 사용함 |

---

# 6. Limitations and Future Work

## 6.1 Limitations

본 프로젝트의 한계는 다음과 같다.

### 1. Validation Loss 미구현

현재 실험은 train loss를 중심으로 확인하였다. 따라서 모델이 학습 데이터에 과적합되었는지 판단하기 어렵다.

향후 OpenWebText 5M tokens를 90:10으로 나누어 validation loss를 함께 측정할 필요가 있다.

```
train_data = first 90%
val_data   = last 10%
```

### 2. 데이터 규모의 한계

실제 GPT-2는 훨씬 큰 규모의 WebText corpus로 학습되었다. 반면 본 프로젝트는 Colab 환경의 한계로 OpenWebText 중 5M tokens만 사용하였다.

### 3. 모델 크기의 한계

본 프로젝트의 모델은 GPT-2 small보다 훨씬 작다.

```
GPT-2 small: emb_dim=768, num_layers=12, num_heads=12
Our model:   emb_dim=256, num_layers=6,  num_heads=8
```

따라서 생성 문장의 자연스러움과 장기 문맥 처리 능력에는 한계가 있다.

### 4. Context Length의 한계

GPT-2 small은 context length 1024를 사용하지만, 본 프로젝트는 `block_size=256`을 사용하였다.

따라서 더 긴 문맥을 반영하는 데 한계가 있다.

### 5. Pretrained Weight 미사용

본 프로젝트는 GPT-2 pretrained weights를 사용하지 않고, 모델 parameter를 처음부터 학습하였다.

따라서 같은 GPT-2 tokenizer를 사용하더라도 실제 GPT-2와 같은 성능을 기대하기는 어렵다.

## 6.2 Future Work

향후 개선 방향은 다음과 같다.

```
1. OpenWebText 5M tokens를 train/validation으로 split하여 validation loss 측정
2. max_tokens를 10M 또는 20M으로 확장
3. block_size를 256에서 512로 확장
4. gradient accumulation을 적용해 effective batch size 증가
5. 학습된 모델 weight 저장 및 재사용
6. OpenWebText pretraining 후 Shakespeare fine-tuning 수행
7. sampling 결과를 temperature와 top_k별로 비교
8. GPT-2 small 구조와 더 가까운 모델 설정 실험
```

---

# 7. How to Run

## 7.1 How to Run

본 프로젝트는 `mini-gpt.py` 파일을 실행하는 방식으로 구성되어 있다.

```
git clone https://github.com/ksm12030-sudo/mini-gpt2.git
cd mini-gpt2
pip install-r requirements.txt
python mini-gpt.py
```

실행하면 tokenizer를 이용해 text corpus를 token id로 변환하고, GPT-style mini language model을 학습한 뒤 sample text를 생성한다. 

## 7.2 File Structure
파일 구조는 다음과 같다 : 
''' mini-gpt2/
│
├── README.md
├── mini-gpt.py
├── requirements.txt
└── training_loss.png '''

