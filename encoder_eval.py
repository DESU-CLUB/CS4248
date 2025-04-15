#Take the llama encoder and evaluate it on the test set

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import os
import random
import numpy as np
from llama_encoder import LlamaEncoder  # Import from llama_encoder.py
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import re

# Set seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Custom dataset class
class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text, 
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Remove the batch dimension
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'text': text
        }

# Function to infer a category from text
def infer_category(text):
    # Simple categorization based on text features
    text = text.lower()
    if "emoji" in text:
        return "emoji_mentioned"
    elif len(re.findall(r'[^\w\s,]', text)) > 5:  # Many non-alphanumeric chars
        return "emoji_heavy"
    elif len(text.split()) < 10:
        return "short_text"
    else:
        return "regular_text"

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load the dataset
    ds = load_dataset("DESUCLUB/combined_emoji_data")
    texts = ds["train"]['text']
    
    # Use a subsample of the dataset
    subsample_size = 50  # Increased for better visualization
    print(f"Using subsample of {subsample_size} examples")
    
    # Take a random subsample
    indices = random.sample(range(len(texts)), subsample_size)
    texts_subsample = [texts[i] for i in indices]
    
    # Initialize tokenizer for Llama
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset and dataloader
    dataset = TextDataset(texts_subsample, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    
    # Initialize the Llama encoder
    encoder = LlamaEncoder(model_name)
    encoder.to(device)
    print(f"Initialized LlamaEncoder with model: {model_name}")
    
    # Process the subsample
    print("Processing subsample...")
    encoder.eval()  # Set to evaluation mode
    
    outputs = []
    texts_processed = []
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        texts_batch = batch['text']
        
        # Get encoder outputs
        with torch.inference_mode():
            encoder_output = encoder(input_ids)
            
            # Get the average of all token embeddings for a sequence embedding
            # This is required since the imported LlamaEncoder returns all token embeddings
            # We need a single embedding vector per sequence
            mask = attention_mask.unsqueeze(-1)
            sequence_embeddings = (encoder_output * mask).sum(dim=1) / mask.sum(dim=1)
            
        # Store outputs and corresponding texts
        outputs.append(sequence_embeddings.cpu())
        texts_processed.extend(texts_batch)
        
        # Print the shape of the encoder output for the first batch
        if len(outputs) == 1:
            print(f"Raw encoder output shape: {encoder_output.shape}")
            print(f"Sequence embedding shape: {sequence_embeddings.shape}")
    
    # Concatenate all outputs
    all_outputs = torch.cat(outputs, dim=0)
    
    print(f"Total number of examples processed: {len(texts_processed)}")
    print(f"Final output tensor shape: {all_outputs.shape}")
    
    # Print a few examples with their embeddings (just the first 5 values)
    for i in range(min(3, len(texts_processed))):
        text = texts_processed[i]
        embedding = all_outputs[i]
        print(f"\nExample {i+1}: {text[:100]}...")
        print(f"Embedding (first 5 values): {embedding[:5]}")
    
    # Save the encoder outputs for future use
    os.makedirs("encoder_outputs", exist_ok=True)
    output_path = f"encoder_outputs/llama_subsample_{subsample_size}.pt"
    torch.save({
        'outputs': all_outputs,
        'texts': texts_processed
    }, output_path)
    print(f"Saved encoder outputs to: {output_path}")
    
    # Visualize the embeddings using PCA
    print("Creating PCA visualization...")
    
    # Convert embeddings to numpy for PCA
    embedding_matrix = all_outputs.numpy()
    
    # Apply PCA to reduce dimensions to 3
    pca = PCA(n_components=3)
    vis_dims = pca.fit_transform(embedding_matrix)
    
    # Create a DataFrame for easier data handling
    samples = pd.DataFrame({
        'text': texts_processed,
        'embed_vis': vis_dims.tolist()
    })
    
    # Infer categories for each sample
    samples['category'] = samples['text'].apply(infer_category)
    categories = samples['category'].unique()
    
    # Create 3D visualization
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap("tab20")
    
    # Plot each sample category individually
    for i, cat in enumerate(categories):
        sub_matrix = np.array(samples[samples["category"] == cat]["embed_vis"].tolist())
        x = sub_matrix[:, 0]
        y = sub_matrix[:, 1]
        z = sub_matrix[:, 2]
        colors = [cmap(i/len(categories))] * len(sub_matrix)
        ax.scatter(x, y, zs=z, zdir='z', c=colors, label=cat)
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title(f'3D PCA of Llama Encoder Embeddings (n={subsample_size})')
    ax.legend(bbox_to_anchor=(1.1, 1))
    
    # Save the figure
    plt.tight_layout()
    fig_path = f"encoder_outputs/llama_embeddings_pca_{subsample_size}.png"
    plt.savefig(fig_path, dpi=300)
    print(f"PCA visualization saved to: {fig_path}")
    
    # Display plot
    plt.show()

if __name__ == "__main__":
    main()

