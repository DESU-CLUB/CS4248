import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Embedding, TransformerEncoder, TransformerEncoderLayer, Linear
from tqdm import tqdm
from transformers import BertTokenizer
from matplotlib import pyplot as plt
from datasets import load_dataset
from llama_encoder import LlamaEncoder
import numpy as np
torch.manual_seed(0) # for reproducibility
torch.set_default_dtype(torch.float32)

# use GPU if available
if torch.cuda.is_available():
   device = torch.device("cuda")
   print('GPU')
elif torch.backends.mps.is_available(): # Apple GPU
   device = torch.device("mps")
   print('MPS')
else:
   device = torch.device("cpu")
   print("CPU")
   
class Encoder(torch.nn.Module):
    def __init__(self, voc_size, embed_size, num_heads = 8, num_layers = 6, device = device):
        super().__init__()
        self.emb = Embedding(voc_size, embed_size, device=device) # (batch, seq_len) -> (batch, seq_len, embed_size)
        enc_layer = TransformerEncoderLayer(embed_size, num_heads, device=device) # (seq_len, batch, embed_size) to same
        self.transformer_enc = TransformerEncoder(enc_layer,num_layers) # (seq_len, batch, embed_size) to same
        self.pred = Linear(embed_size, 2048) #  (seq_len, batch, embed_size) -> (seq_len, batch, 1)
   
    def forward(self, x):
        x = self.emb(x).permute(1, 0, 2) # (batch, seq_len) -> (seq_len, batch, embed_size)
        x = self.transformer_enc(x).permute(1, 0, 2) # (seq_len, batch, embed_size) -> (batch, seq_len, embed_size)
        return self.pred(x.mean(dim=1)).squeeze(1) # (batch, seq_len, embed_size) -> (batch)

epochs = 100
lr = 1e-3
batch_size = 100

# dataset definition and preprocessing
ds = load_dataset("KomeijiForce/Text2Emoji")
X = ds["train"]['text']
y = ds['train']['emoji'] #ds["train"]['labels']
label_encoder = LlamaEncoder("meta-llama/Meta-Llama-3.2-1B-Instruct")
# Freeze the LlamaEncoder parameters to prevent training
for param in label_encoder.parameters():
    param.requires_grad = False

# Ensure the encoder is in evaluation mode
label_encoder.eval()
print("LlamaEncoder has been frozen and set to evaluation mode")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

X_train = [str(x) for x in X_train]
X_test = [str(x) for x in X_test]

X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, return_tensors="pt")

voc_size = tokenizer.vocab_size
embed_size = 128
num_heads = 8
num_layers = 6

train_dataset = TensorDataset(X_train_tokenized['input_ids'], 
                              torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(X_test_tokenized['input_ids'], 
                              torch.tensor(y_test, dtype=torch.float32))

model = Encoder(voc_size, embed_size, num_heads, num_layers, device).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# train
losses = []
loss_fn = torch.nn.MSELoss()
training_loop = tqdm(range(epochs)) 
for epoch in training_loop:
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device) 
        y_batch = y_batch.to(device)
        with torch.no_grad():
            processed_y_batch = label_encoder.encode(y_batch)
        optimizer.zero_grad()
        print(model(X_batch).shape)
        print(processed_y_batch.shape)
        assert 0
        loss = loss_fn(model(X_batch), processed_y_batch)
        loss.backward()
        optimizer.step()     
    training_loop.set_postfix(loss = loss.item())
    scheduler.step()
    losses.append(loss.item())
plt.plot(losses, label='Training loss')
plt.show()

model.eval()
with torch.no_grad():
    y_train_pred = model(torch.tensor(X_train, dtype=torch.long).to(device))
    y_test_pred = model(torch.tensor(X_test, dtype=torch.long).to(device))
    print("Train MSE: ", loss_fn(y_train_pred, torch.tensor(y_train, dtype=torch.float32).to(device)).item())
    print("Test MSE: ", loss_fn(y_test_pred, torch.tensor(y_test, dtype=torch.float32).to(device)).item())