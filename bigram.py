import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

#---------------


torch.manual_seed(1337)

# dataset to train on. !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# unique characters that occur in dataset
chars = sorted(list(set(text)))
vocab_size= len(chars)

# mapping of characters from integers

stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
encode = lambda s: [stoi[c] for c in s] # encode : takes string give list of inetgers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder : takes a list of inetegers gives string

# encoding dataset
data = torch.tensor(encode(text), dtype=torch.long)

# data splitting for training and validation
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1: i+block_size+1] for i in ix])
  return x, y

@torch.no_grad()
def estimate_loss():
  out = {}
  model.eval()
  for split in ['train', 'val']:
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
      X, Y = get_batch(split)
      logits, loss = model(X, Y)
      losses[k] = loss.iten()
    out(split) = losses.mean()
  model.train()
  return out

class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx) # (B, T, c) in this case it is going to arrange it 4x8x65

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
      # get the predictions
      logits, loss = self(idx)
      # focus only on the last time step
      logits = logits[:, -1, :] # become (B, T)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # (B, T)
      # sample from the distribution
      idx_next = torch.multinomial(probs, num_samples=1)
      # append sampled index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# optimiser
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

for iter in range(max_iters):

  # every once in a while it will evaluate loss on train and val sets
  if iter % eval_interval == 0:
    losses = estimate_loss()
    print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  # sample the batch of data
  xb, yb = get_batch('train')

  # loss evaluation
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

context = torch.zeros((1, 1), dtype = torch.long, device=device)
print(decode(m.generate(context, max_new_token=500)[0].tolist()))