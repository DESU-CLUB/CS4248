import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel, AutoTokenizer
from datasets import load_from_disk
from torch.utils.data import Dataset, DataLoader
import random
from tqdm import tqdm
from llama_encoder import LlamaEncoder
from sklearn.preprocessing import normalize

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Custom dataset class for text with topics
class TopicDataset(Dataset):
    def __init__(self, texts, topics, tokenizer, max_length=128):
        self.texts = texts
        self.topics = topics
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        topic = self.topics[idx]
        
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
            'text': text,
            'topic': topic
        }

# Function to get embeddings from BERT
def get_bert_embeddings(texts, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = BertModel.from_pretrained('bert-base-multilingual-cased')
    model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    class SimpleTextDataset(Dataset):
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
            
            return {
                'input_ids': encoded['input_ids'].squeeze(),
                'attention_mask': encoded['attention_mask'].squeeze(),
                'text': text
            }
    
    dataset = SimpleTextDataset(texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Process texts in batches
    all_embeddings = []
    all_texts = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing BERT embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts_batch = batch['text']
            
            # Get BERT outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Use the [CLS] token embedding as the sentence embedding
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            
            all_embeddings.append(embeddings)
            all_texts.extend(texts_batch)
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    return all_embeddings, all_texts

# Function to get embeddings from our custom encoder
def get_custom_embeddings(texts, topics=None, batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # Initialize BERT tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Load a BERT model with our finetuned weights
    try:
        # First load the base BERT model
        model = BertModel.from_pretrained('bert-base-multilingual-cased')
        
        # Load our finetuned weights
        try:
            # Try loading from local file first
            state_dict = torch.load('tuned_bert_encoder_model.pt', map_location=device)
            print("Loading finetuned BERT weights from local file")
        except FileNotFoundError:
            # If local file not found, try downloading from HuggingFace
            state_dict = torch.hub.load_state_dict_from_url(
                "https://huggingface.co/DESUCLUB/emoji_encoder_elco/resolve/main/tuned_bert_encoder_model.pt",
                map_location=device
            )
            print("Loading finetuned BERT weights from HuggingFace")
            
        # Load the state dict into the model
        model.load_state_dict(state_dict)
        print("Successfully loaded finetuned BERT weights")
        
        # Use the BERT tokenizer
        tokenizer = bert_tokenizer
        using_bert = True
    except Exception as e:
        print(f"Could not load finetuned BERT, using Llama encoder instead: {e}")
        # If we fall back to Llama, use the Llama tokenizer
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = LlamaEncoder(model_name)
        using_bert = False
    
    model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    if topics is not None:
        dataset = TopicDataset(texts, topics, tokenizer)
    else:
        dataset = TopicDataset(texts, ["unknown"] * len(texts), tokenizer)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Process texts in batches
    all_embeddings = []
    all_texts = []
    all_topics = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing custom embeddings"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            texts_batch = batch['text']
            topics_batch = batch['topic']
            
            # Different handling based on the model type
            if using_bert:
                # Get BERT outputs
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                # Use the [CLS] token embedding (first token) as the sentence representation
                sequence_embeddings = outputs.last_hidden_state[:, 0, :]
            else:
                # For Llama encoder
                encoder_output = model(input_ids)
                mask = attention_mask.unsqueeze(-1)
                sequence_embeddings = (encoder_output * mask).sum(dim=1) / mask.sum(dim=1)
            
            all_embeddings.append(sequence_embeddings.cpu().numpy())
            all_texts.extend(texts_batch)
            all_topics.extend(topics_batch)
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    
    return all_embeddings, all_texts, all_topics

# Function to visualize embeddings in 3D space
def visualize_embeddings(embeddings, topics, title, output_path, limit=None):
    if limit and len(embeddings) > limit:
        # Sample a subset
        indices = random.sample(range(len(embeddings)), limit)
        embeddings = embeddings[indices]
        topics = [topics[i] for i in indices]
    
    # Apply PCA to reduce dimensions to 3
    pca = PCA(n_components=3)
    vis_dims = pca.fit_transform(embeddings)
    
    # Create a DataFrame for easier data handling
    samples = pd.DataFrame({
        'topic': topics,
        'embed_vis': list(vis_dims)
    })
    
    # Get unique topics
    unique_topics = sorted(set(topics))
    
    # Create 3D visualization
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    cmap = plt.get_cmap("tab20")
    
    # Plot each topic individually
    for i, topic in enumerate(unique_topics):
        sub_matrix = np.array([sample['embed_vis'] for _, sample in samples[samples['topic'] == topic].iterrows()])
        
        if len(sub_matrix) > 0:  # Make sure we have samples for this topic
            x = sub_matrix[:, 0]
            y = sub_matrix[:, 1]
            z = sub_matrix[:, 2]
            colors = [cmap(i % 20 / 20)] * len(sub_matrix)
            ax.scatter(x, y, zs=z, zdir='z', c=colors, label=topic, alpha=0.7)
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title(title)
    
    # Create a separate legend that's part of the figure but not overlapping with the plot
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend
    leg = fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), 
                     ncol=max(1, len(unique_topics) // 20), frameon=True, fontsize='small')
    
    # Save the figure
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.close(fig)
    
    # Also create a 2D visualization
    create_2d_plot(vis_dims, topics, title, output_path.replace('.png', '_2d.png'))
    
    return vis_dims

# Function to create 2D plot
def create_2d_plot(vis_dims, topics, title, output_path):
    # Create a DataFrame for easier data handling
    samples = pd.DataFrame({
        'topic': topics,
        'x': vis_dims[:, 0],
        'y': vis_dims[:, 1]
    })
    
    # Get unique topics
    unique_topics = sorted(set(topics))
    
    # Create 2D visualization with a larger figure to accommodate the legend
    fig, ax = plt.subplots(figsize=(15, 10))
    cmap = plt.get_cmap("tab20")
    
    # Plot each topic individually
    for i, topic in enumerate(unique_topics):
        topic_data = samples[samples['topic'] == topic]
        if len(topic_data) > 0:
            ax.scatter(topic_data['x'], topic_data['y'], 
                      c=[cmap(i % 20 / 20)] * len(topic_data),
                      label=topic, alpha=0.7)
    
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_title(f"{title} (2D)")
    
    # Create a separate legend that's part of the figure but not overlapping with the plot
    handles, labels = ax.get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0, 0.85, 1])  # Make room for legend
    leg = fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.98, 0.5), 
                    ncol=max(1, len(unique_topics) // 20), frameon=True, fontsize='small')
    
    # Save the figure with the legend
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"2D visualization saved to: {output_path}")
    
    plt.close(fig)

# Function to find nearest examples
def find_closest_examples(embeddings, texts, topics, n=5):
    # Compute cosine similarity matrix
    embeddings_normalized = normalize(embeddings)
    similarity_matrix = cosine_similarity(embeddings_normalized)
    
    # For each example, find the most similar examples
    results = []
    for i in range(len(texts)):
        # Get similarities to current example (exclude self)
        similarities = similarity_matrix[i]
        similarities[i] = -1  # Exclude self
        
        # Get indices of most similar examples
        most_similar_indices = np.argsort(similarities)[-n:][::-1]
        
        # Collect results
        result = {
            'text': texts[i],
            'topic': topics[i],
            'similar_examples': [
                {
                    'text': texts[idx],
                    'topic': topics[idx],
                    'similarity': similarities[idx]
                }
                for idx in most_similar_indices
            ]
        }
        results.append(result)
    
    return results

# Main function
def main(n_examples=500, limit_vis=300):
    set_seed(42)
    
    # Create output directory
    os.makedirs("embedding_visualizations", exist_ok=True)
    
    # Load the dataset with topics
    print("Loading dataset with topics...")
    try:
        dataset = load_from_disk("combined_emoji_data_with_topics")
        print(f"Loaded dataset with {len(dataset)} examples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Get a sample of the dataset for analysis
    if n_examples < len(dataset):
        indices = random.sample(range(len(dataset)), n_examples)
        subset = dataset.select(indices)
    else:
        subset = dataset
    
    texts = subset["text"]
    topics = subset["topic"]
    
    print(f"Working with {len(texts)} examples across {len(set(topics))} topics")
    
    # Get BERT embeddings
    print("\nComputing BERT embeddings...")
    bert_embeddings, bert_texts = get_bert_embeddings(texts)
    
    # Get custom encoder embeddings
    print("\nComputing custom encoder embeddings...")
    custom_embeddings, custom_texts, custom_topics = get_custom_embeddings(texts, topics)
    
    # Create visualizations
    print("\nCreating visualizations...")
    bert_vis_dims = visualize_embeddings(
        bert_embeddings, 
        topics, 
        f"BERT Embeddings by Topic (n={len(bert_embeddings)})",
        "embedding_visualizations/bert_topic_embeddings.png",
        limit=limit_vis
    )
    
    custom_vis_dims = visualize_embeddings(
        custom_embeddings, 
        topics, 
        f"Custom Encoder Embeddings by Topic (n={len(custom_embeddings)})",
        "embedding_visualizations/custom_topic_embeddings.png",
        limit=limit_vis
    )
    
    # Find examples with different topic assignments
    print("\nFinding interesting examples for comparison...")
    bert_similar = find_closest_examples(bert_embeddings, texts, topics)
    custom_similar = find_closest_examples(custom_embeddings, texts, topics)
    
    # Find examples that have very different nearest neighbors between encoders
    comparison_examples = []
    
    for i in range(min(10, len(bert_similar))):
        example = {
            'text': texts[i],
            'topic': topics[i],
            'bert_neighbors': [
                {'text': ex['text'], 'topic': ex['topic']} 
                for ex in bert_similar[i]['similar_examples']
            ],
            'custom_neighbors': [
                {'text': ex['text'], 'topic': ex['topic']} 
                for ex in custom_similar[i]['similar_examples']
            ]
        }
        
        # Check if the encoders assign different topics to nearest neighbors
        bert_neighbor_topics = set(n['topic'] for n in example['bert_neighbors'])
        custom_neighbor_topics = set(n['topic'] for n in example['custom_neighbors'])
        
        if bert_neighbor_topics != custom_neighbor_topics:
            comparison_examples.append(example)
    
    # Save comparison results
    with open("embedding_visualizations/encoder_comparison.txt", "w") as f:
        f.write("Comparison of BERT and Custom Encoder Topic Assignments\n")
        f.write("=" * 80 + "\n\n")
        
        for i, example in enumerate(comparison_examples[:5]):
            f.write(f"Example {i+1}: '{example['text']}'\n")
            f.write(f"Topic: {example['topic']}\n\n")
            
            f.write("BERT nearest neighbors:\n")
            for j, neighbor in enumerate(example['bert_neighbors']):
                f.write(f"  {j+1}. Topic: {neighbor['topic']}\n")
                f.write(f"     Text: {neighbor['text'][:100]}...\n")
            
            f.write("\nCustom Encoder nearest neighbors:\n")
            for j, neighbor in enumerate(example['custom_neighbors']):
                f.write(f"  {j+1}. Topic: {neighbor['topic']}\n")
                f.write(f"     Text: {neighbor['text'][:100]}...\n")
            
            f.write("\n" + "-" * 80 + "\n\n")
    
    print(f"Saved comparison results to embedding_visualizations/encoder_comparison.txt")
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main(n_examples=500)  # Adjust the number of examples as needed 