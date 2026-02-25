import torch

# Load text
with open(r"C:\Users\shrey\OneDrive\Documents\coding_shit\projects\mini-gpt\data\input.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("Length of dataset:", len(text))

chars = sorted(list(set(text)))
vocab_size = len(chars)
print("Vocab size:" , vocab_size)

stoi = {}
i = 0
for ch in chars:
    stoi[ch] = i
    i += 1
i = 0
itos = {}
for ch in chars:
    itos[i] = ch
    i += 1

#stoi = {ch: i for i, ch in enumerate(chars)}
#itos = {i: ch for ch, i in stoi.items()}

len_train = int(0.9 * len(text))
train = text[:len_train]
val = text[len_train:]

block_size = 128
batch_size = 4

def get_batch(split):
    data_source = train if split == "train" else val
    
    ix = torch.randint(0, len(data_source) - block_size, (batch_size,))
    
    x_list = []
    y_list = []
    
    for i in ix:
        x_str = data_source[i:i+block_size]
        y_str = data_source[i+1:i+block_size+1]
        
        x_int = torch.tensor([stoi[c] for c in x_str], dtype=torch.long)
        y_int = torch.tensor([stoi[c] for c in y_str], dtype=torch.long)
        
        x_list.append(x_int)
        y_list.append(y_int)
    
    x = torch.stack(x_list)  # shape (batch_size, block_size)
    y = torch.stack(y_list)  # shape (batch_size, block_size)
    
    return x, y