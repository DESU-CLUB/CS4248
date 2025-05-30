import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, AutoTokenizer
from tqdm import tqdm
from matplotlib import pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset
from llama_encoder import LlamaEncoder
torch.manual_seed(0) # for reproducibility
torch.set_default_dtype(torch.float32)
from huggingface_hub import login

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

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

epochs = 3
lr = 1e-5
batch_size = 20

label_encoder = LlamaEncoder("meta-llama/Llama-3.2-1B-Instruct")
label_encoder.to(device)
# Freeze the LlamaEncoder parameters to prevent training
for param in label_encoder.parameters():
    param.requires_grad = False

# Ensure the encoder is in evaluation mode
label_encoder.eval()
print("LlamaEncoder has been frozen and set to evaluation mode")

# Linear layer to match Llama output dimension with Bert
proj_layer = torch.nn.Linear(2048, 768).to(device)

class EncoderDataset(Dataset):
    def __init__(self, X, attention_mask, y):
        self.X = X
        self.attention_mask = attention_mask
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        with torch.no_grad():
            return self.X[idx].clone().detach(), self.attention_mask[idx].clone().detach(), self.y[idx]
            
# dataset definition and preprocessing
ds = load_dataset("KomeijiForce/Text2Emoji")
X = ds["train"]['text']
y = ds['train']['emoji'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
label_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
if label_tokenizer.pad_token is None:
    label_tokenizer.pad_token = "<|finetune_right_pad_id|>"
X_train = [str(x) for x in X_train]
X_test = [str(x) for x in X_test]

print("Tokenizing training data...")
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
y_train = label_tokenizer(y_train, padding=True, truncation=True, return_tensors="pt").input_ids
print("Tokenizing test data...")
X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, return_tensors="pt")
y_test = label_tokenizer(y_test, padding=True, truncation=True, return_tensors="pt").input_ids

train_dataset = EncoderDataset(X_train_tokenized['input_ids'], 
                               X_train_tokenized['attention_mask'],
                               y_train)
test_dataset = EncoderDataset(X_test_tokenized['input_ids'], 
                              X_test_tokenized['attention_mask'],
                              y_test)

model = BertModel.from_pretrained("bert-base-multilingual-cased").to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print("Training dataset size: ", len(train_dataset))
print("Test dataset size: ", len(test_dataset))
print("Start training...")

# train
losses = []
loss_fn = torch.nn.MSELoss()
training_loop = tqdm(range(epochs))
model.train() 
for epoch in training_loop:
    print('Epoch: ', epoch)
    for X_batch, attention_mask_batch, y_batch in loader:
        X_batch = X_batch.to(device) 
        attention_mask_batch = attention_mask_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            y_batch = label_encoder(y_batch)
            y_batch = y_batch.mean(dim=1)
            y_batch = proj_layer(y_batch)  # Project Llama output to match BERT dimensions
        loss = loss_fn(model(X_batch, attention_mask = attention_mask_batch).last_hidden_state.mean(dim=1), y_batch)
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()     
    training_loop.set_postfix(loss = loss.item())
    scheduler.step()
    losses.append(loss.item())
    torch.save(model.state_dict(), 'tuned_bert_encoder_model.pt')
plt.plot(losses, label='Training loss')
plt.show()

model.eval()
with torch.no_grad():
    train_losses = []
    test_losses = []
    
    # Process training data in batches
    for i in range(0, len(X_train_tokenized['input_ids']), batch_size):
        end_idx = min(i + batch_size, len(X_train_tokenized['input_ids']))
        batch_input_ids = torch.tensor(X_train_tokenized['input_ids'][i:end_idx], dtype=torch.long).to(device)
        batch_attention_ids = torch.tensor(X_train_tokenized['attention_mask'][i:end_idx], dtype=torch.long).to(device)
        batch_labels = y_train[i:end_idx].to(device)
        
        batch_pred = model(batch_input_ids, attention_mask = batch_attention_ids).last_hidden_state.mean(dim=1)
        batch_loss = loss_fn(batch_pred, proj_layer(label_encoder(batch_labels).mean(dim=1))).item()
        train_losses.append(batch_loss)
        torch.cuda.empty_cache()
    
    # Process test data in batches
    for i in range(0, len(X_test_tokenized['input_ids']), batch_size):
        end_idx = min(i + batch_size, len(X_test_tokenized['input_ids']))
        batch_input_ids = torch.tensor(X_test_tokenized['input_ids'][i:end_idx], dtype=torch.long).to(device)
        batch_attention_ids = torch.tensor(X_test_tokenized['attention_mask'][i:end_idx], dtype=torch.long).to(device)
        batch_labels = y_test[i:end_idx].to(device)
        
        batch_pred = model(batch_input_ids, attention_mask = batch_attention_ids).last_hidden_state.mean(dim=1)
        batch_loss = loss_fn(batch_pred, proj_layer(label_encoder(batch_labels).mean(dim=1))).item()
        test_losses.append(batch_loss)
        torch.cuda.empty_cache()
    
    # Calculate and print average losses
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_test_loss = sum(test_losses) / len(test_losses)
    
    print(f"Train MSE: {avg_train_loss:.6f}")
    print(f"Test MSE: {avg_test_loss:.6f}")

    # Save the trained model
    # Save model locally
    torch.save(model.state_dict(), 'tuned_bert_encoder_model.pt')
    print("Model saved successfully to 'tuned_bert_encoder_model.pt'")
