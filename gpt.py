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
    def __init__(self, X, attention, y):
        self.input_ids = X
        self.attention_mask = attention
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.attention_mask[idx], self.y[idx]

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
y_tokenized = tokenizer(y, padding=True, truncation=True, return_tensors="pt")
# tokenized_dataset = TokenizedDataset(y_tokenized)
# y_loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

# embeddings = list()
# with torch.no_grad():
#     for y_batch_tokenized in y_loader:
#         out = encoder(input_ids=y_batch_tokenized['input_ids'].to(device), attention_mask=y_batch_tokenized['attention_mask'].to(device))
#         embeddings_batch = out.last_hidden_state
#         embeddings.append(embeddings_batch)
#         torch.cuda.empty_cache()
# X = torch.cat(embeddings,dim=0)
label_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
if label_tokenizer.pad_token is None:
    label_tokenizer.pad_token = "<|finetune_right_pad_id|>"
label_tokenized = label_tokenizer(y, padding=True, truncation = True, return_tensors="pt")
label_seq_length = label_tokenized['input_ids'].size(1)

X_train, X_test, X_attention_mask_train, X_attention_mask_test, y_train, y_test = train_test_split(y_tokenized['input_ids'], y_tokenized['attention_mask'], label_tokenized['input_ids'], test_size = 0.2, random_state = 0)

train_dataset = DecoderDataset(X_train, X_attention_mask_train, y_train)
test_dataset = DecoderDataset(X_test, X_attention_mask_test, y_test)

model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# train
print("Start training")
losses = []
training_loop = tqdm(range(epochs))
model.train() 
for epoch in training_loop:
    print('Epoch: ', epoch)
    for X_batch, X_attention_mask, y_batch in loader:
        X_batch = X_batch.to(device)
        attention_mask = X_attention_mask.to(device)
        with torch.no_grad():
            input_embeddings = encoder(input_ids = X_batch, attention_mask = attention_mask).last_hidden_state
        input_embeddings = input_embeddings[:, :label_seq_length]
        if input_embeddings.size(1) < label_seq_length:
            padding = torch.zeros(input_embeddings.size(0), label_seq_length - input_embeddings.size(1), input_embeddings.size(2)).to(device)
            input_embeddings = torch.cat([input_embeddings, padding], dim=1)
        
        y_batch = y_batch.to(device)
        optimizer.zero_grad()

        out = model(inputs_embeds=input_embeddings.to(device), labels=y_batch)
        loss = out.loss
        loss.backward()
        optimizer.step()
        torch.cuda.empty_cache()     
    training_loop.set_postfix(loss = loss.item())
    scheduler.step()
    losses.append(loss.item())
    torch.save(model.state_dict(), 'tuned_gpt2_decoder_model.pt')

print("Start eval")
model.eval()
with torch.no_grad():
    train_losses = []
    test_losses = []
    
    # Process training data in batches
    for i in range(0, len(X_train), batch_size):
        end_idx = min(i + batch_size, len(X_train))
        input_ids = X_train[i:end_idx].to(device)
        attention_mask = X_attention_mask_train[i:end_idx].to(device)
        with torch.no_grad():
            input_embeddings = encoder(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state
        input_embeddings = input_embeddings[:, :label_seq_length]
        if input_embeddings.size(1) < label_seq_length:
            padding = torch.zeros(input_embeddings.size(0), label_seq_length - input_embeddings.size(1), input_embeddings.size(2)).to(device)
            input_embeddings = torch.cat([input_embeddings, padding], dim=1)
        batch_labels = y_train[i:end_idx].to(device)
        
        batch_out = model(inputs_embeds = input_embeddings.to(device), labels = batch_labels)
        batch_loss = batch_out.loss.item()
        train_losses.append(batch_loss)
        torch.cuda.empty_cache()
    
    # Process test data in batches
    for i in range(0, len(X_test), batch_size):
        end_idx = min(i + batch_size, len(X_test))
        input_ids = X_test[i:end_idx].to(device)
        attention_mask = X_attention_mask_test[i:end_idx].to(device)
        with torch.no_grad():
            input_embeddings = encoder(input_ids = input_ids, attention_mask = attention_mask).last_hidden_state
        input_embeddings = input_embeddings[:, :label_seq_length]
        if input_embeddings.size(1) < label_seq_length:
            padding = torch.zeros(input_embeddings.size(0), label_seq_length - input_embeddings.size(1), input_embeddings.size(2)).to(device)
            input_embeddings = torch.cat([input_embeddings, padding], dim=1)
        batch_labels = y_test[i:end_idx].to(device)
        
        batch_out = model(inputs_embeds = input_embeddings.to(device), labels = batch_labels)
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


