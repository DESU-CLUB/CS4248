import torch
from transformers import BertTokenizer, BertModel
torch.manual_seed(0) # for reproducibility
torch.set_default_dtype(torch.float32)
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np

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
   
encoder = BertModel.from_pretrained('bert-base-multilingual-cased')
state_dict = torch.load('tuned_bert_encoder_model.pt', map_location="cpu")
encoder.load_state_dict(state_dict)
encoder.to(device)

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
ds = [
    # Gendered terms
    'man', 'woman', #'male', 'female', 'he', 'she', 'him', 'her',

    # Neutral professions
    #'doctor', 'nurse', 'teacher', 'engineer', 'scientist', 'pilot', 'lawyer', 'janitor',

    # Gender-stereotypical professions
    'actor', 'actress', 'policeman', 'policewoman', 'fireman', 'firewoman', 'flight attendant', 'stewardess',

    # Neutral adjectives
    #'strong', 'weak', 'intelligent', 'emotional', 'logical', 'caring', 'brave', 'nurturing'
]
ds_tokenized = tokenizer(ds, padding=True, truncation=True, return_tensors="pt")

y = encoder(input_ids = ds_tokenized['input_ids'].to(device), attention_mask = ds_tokenized["attention_mask"].to(device)).last_hidden_state

# Extract the embeddings (mean pooling across tokens for simplicity)
# Shape of `y` is (batch_size, seq_len, hidden_dim)
embeddings = y.mean(dim=1).detach().cpu().numpy()  # Shape: (batch_size, hidden_dim)

# Reduce dimensions to 2D using PCA or t-SNE
# Option 1: PCA
pca = PCA(n_components=2)
reduced_embeddings = pca.fit_transform(embeddings)

# Option 2: t-SNE (uncomment to use t-SNE instead of PCA)
# tsne = TSNE(n_components=2, random_state=42)
# reduced_embeddings = tsne.fit_transform(embeddings)

# Labels for the embeddings
labels = ds  # ['man', 'black', 'white', 'yellow', 'red']

# Plot the 2D embeddings
plt.figure(figsize=(8, 6))
for i, label in enumerate(labels):
    x, y = reduced_embeddings[i]
    plt.scatter(x, y, label=label)
    plt.text(x + 0.02, y + 0.02, label, fontsize=9)  # Add labels near points

plt.title("2D Visualization of Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.legend()
plt.grid(True)
plt.show()