import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import Dataset
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

class TokenizedDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }
        
class DecoderDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        with torch.no_grad():
            return self.X[idx].clone().detach(), self.y[idx]

epochs = 3
lr = 1e-5
batch_size = 20
        
encoder = BertModel.from_pretrained('bert-base-multilingual-cased')
state_dict = torch.load('tuned_bert_encoder_model.pt', map_location="cpu")
encoder.load_state_dict(state_dict)
encoder.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            
# dataset definition and preprocessing
ds = load_dataset("KomeijiForce/Text2Emoji")
y = ds["train"]['text']
y = [str(i) for i in ds["train"]['text']]
y_tokenized = tokenizer(y, padding=True, truncation=True, return_tensors="pt")[:20]
tokenized_dataset = TokenizedDataset(y_tokenized)
y_loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

embeddings = list()
with torch.no_grad():
    for y_batch_tokenized in y_loader:
        out = encoder(input_ids=y_batch_tokenized['input_ids'].to(device), attention_mask=y_batch_tokenized['attention_mask'].to(device))
        embeddings_batch = out.last_hidden_state
        embeddings.append(embeddings_batch)
X = torch.cat(embeddings,dim=0)
print(X.shape)
label_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if label_tokenizer.pad_token is None:
    label_tokenizer.pad_token = "<|finetune_right_pad_id|>"
label_tokenized = label_tokenizer(y, padding=True, truncation = True, return_tensors="pt")[:20]

label_seq_length = label_tokenized['input_ids'].size(1)
X = X[:, :label_seq_length, :]
if X.size(1) < label_seq_length:
    padding = torch.zeros(X.size(0), label_seq_length - X.size(1), X.size(2)).to(device)
    X = torch.cat([X, padding], dim=1)

X_train, X_test, y_train, y_test = train_test_split(X, label_tokenized['input_ids'], test_size = 0.2, random_state = 0)

train_dataset = DecoderDataset(X_train, y_train)
test_dataset = DecoderDataset(X_test, y_test)

model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# train
losses = []
training_loop = tqdm(range(epochs))
model.train() 
for epoch in training_loop:
    print('Epoch: ', epoch)
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device) 
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        out = model(input_embeds=X_batch, labels=y_batch)
        loss = out.loss
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()     
    training_loop.set_postfix(loss = loss.item())
    scheduler.step()
    losses.append(loss.item())
    torch.save(model.state_dict(), 'tuned_gpt2_decoder_model.pt')

model.eval()
with torch.no_grad():
    train_losses = []
    test_losses = []
    
    # Process training data in batches
    for i in range(0, len(X_train), batch_size):
        end_idx = min(i + batch_size, len(X_train))
        batch_input_ids = X_train[i:end_idx].to(device)
        batch_labels = y_train[i:end_idx].to(device)
        
        batch_out = model(input_embeds = batch_input_ids, labels = batch_labels)
        batch_loss = batch_out.loss.item()
        train_losses.append(batch_loss)
        torch.cuda.empty_cache()
    
    # Process test data in batches
    for i in range(0, len(X_test), batch_size):
        end_idx = min(i + batch_size, len(X_test))
        batch_input_ids = X_test[i:end_idx].to(device)
        batch_labels = y_test[i:end_idx].to(device)
        
        batch_out = model(input_embeds = batch_input_ids, labels = batch_labels)
        batch_loss = batch_out.loss.item()
        test_losses.append(batch_loss)
        torch.cuda.empty_cache()
    
    # Calculate and print average losses
    avg_train_loss = sum(train_losses) / len(train_losses)
    avg_test_loss = sum(test_losses) / len(test_losses)
    
    print(f"Train MSE: {avg_train_loss:.6f}")
    print(f"Test MSE: {avg_test_loss:.6f}")

    # Save the trained model
    # Save model locally
    torch.save(model.state_dict(), 'tuned_gpt2_decoder_model.pt')
    print("Model saved successfully to 'tuned_gpt2_decoder_model.pt'")

# Extract GPT-2 outputs with .logits


