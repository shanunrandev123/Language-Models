import torch.nn as nn
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

batch_size = 32 # independent sequences in parallel
block_size = 8 # max context length to generate predictions

max_iter = 2000

eval_interval = 300

learning_rate = 1e-2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

eval_iters = 200

n_embd = 32

torch.manual_seed(101)



with open('C:/Users/Asus/Downloads/tiny_shakespeare.txt', mode = 'r', encoding='utf-8') as f:
    text = f.read()
    
    
chars = sorted(list(set(text)))

vocab_size = len(chars)

print(''.join(chars))

print(len(chars))


#string to integer mapping
stoi = {ch:i for i,ch in enumerate(chars)}


itos = {i:ch for i,ch in enumerate(chars)}



encode = lambda s : [stoi[c] for c in s]

data = torch.tensor(encode(text), dtype=torch.long)

print(data.shape)
print(data.dtype)
print(data[:100])



#seperate data into train and validation
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


X = train_data[:block_size]
y = train_data[1:block_size + 1]


for t in range(block_size):
    context = X[:t + 1]
    target = y[t]
    
    print(f'when input is {context} the target : {target}')
    
    


def get_batch(split):
    
    data = train_data if split == 'train' else val_data
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    x = torch.stack([data[i:i+block_size] for i in ix])
    
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    
    X = x.to(device)
    y = y.to(device)
    
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = model(X, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

    
    
#(B,T,C) - Batch, time and channel - here batch is batch size which is 4, 
# T is time which is the block size i.e. 8 and 
# C is the number of channels which is vocab size 65



class BigramModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, idx, targets = None):
        
        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C)
        x = tok_emb + pos_emb
        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        logits = self.lm_head(token_embeddings) # (B, T, vocab_size)
        
        if targets is None:
            loss = None
        else:
            
        
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    
    def generate(self, idx, max_tokens):
        
        for _ in range(max_tokens):
            
            #self(idx) is used to call the forward function
            
            
                

            logits, loss = self(idx)


            #focus only on last time stamp
            logits = logits[:, -1, :]

            #apply softmax to get probs

            probs = F.softmax(logits, dim = 1)

            #sample the next token from the batch

            idx_next = torch.multinomial(probs, num_samples = 1)

            idx = torch.cat((idx, idx_next), dim = 1)
            
        return idx
    
    
    
    
    
class Head(nn.Module):
    
    
    def __init__(self, head_size):
        super().__init__()
        
        self.key = nn.Linear(n_embd, head_size, bias=False)
        
        self.query = nn.Linear(n_embd, head_size, bias=False)
        
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        
    
    

model = BigramModel(vocab_size)

m = model.to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr = learning_rate)


for iter in range(max_iter):
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f'Iter {iter} Train loss: {losses["train"]} Val loss: {losses["val"]}')
        
        
    #sample a batch of data
    
    xb, yb = get_batch('train')
    
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    

idx = torch.zeros((1,1), dtype = torch.long)
    
idx1 = m.generate(idx, max_tokens=100)[0].tolist()

print(''.join(itos[i] for i in idx1))
    
        
    
    
    
    
    
    
    
    
