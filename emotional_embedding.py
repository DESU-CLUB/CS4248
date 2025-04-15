import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from transformers import AutoTokenizer
import sys
import os

# Add parent directory to path to import from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the LlamaEncoder from project
from llama_encoder import LlamaEncoder

# Constants
MODEL_NAME = "meta-llama/Llama-3.2-1B-Instruct"
EMOJI_ANALYSIS_FILE = "emoji_analysis.json"
RESULTS_FILE = "emoji_embedding_results.csv"
BATCH_SIZE = 16

def load_emoji_analysis() -> List[Dict[str, Any]]:
    """Load the emoji analysis results from the JSON file"""
    try:
        with open(EMOJI_ANALYSIS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {EMOJI_ANALYSIS_FILE} not found. Run emotion_entailment.py first.")
        return []

def prepare_data_for_encoding(emoji_data: List[Dict[str, Any]]) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Prepare data for encoding: emojis, descriptions, and emotions"""
    emojis = []
    descriptions = []
    emotions = []
    emotion_explanations = []
    
    for item in emoji_data:
        emoji = item["emoji"]
        desc = item["visual_description"]
        
        # Skip if missing data
        if not emoji or not desc:
            continue
            
        emojis.append(emoji)
        descriptions.append(desc)
        
        # Extract top emotions and their explanations
        if "top_emotions" in item and item["top_emotions"]:
            # Primary emotion (first in the list)
            primary_emotion = item["top_emotions"][0]["emotion"]
            emotions.append(primary_emotion)
            
            # Combine all emotion explanations
            all_explanations = " ".join([e["explanation"] for e in item["top_emotions"] if "explanation" in e])
            emotion_explanations.append(all_explanations)
        else:
            emotions.append("")
            emotion_explanations.append("")
    
    return emojis, descriptions, emotions, emotion_explanations

def encode_texts(texts: List[str], encoder: LlamaEncoder, tokenizer: AutoTokenizer) -> torch.Tensor:
    """Encode a list of texts using the LlamaEncoder"""
    embeddings = []
    
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch_texts = texts[i:i+BATCH_SIZE]
            
            # Tokenize
            encodings = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
            
            # Get embeddings
            outputs = encoder(encodings.input_ids)
            
            # Average pooling across token dimension to get sentence embedding
            # Shape: (batch_size, hidden_size)
            batch_embeddings = outputs.mean(dim=1)
            
            embeddings.append(batch_embeddings)
    
    # Concatenate all batches
    return torch.cat(embeddings, dim=0)

def calculate_cosine_similarity(embeddings1: torch.Tensor, embeddings2: torch.Tensor) -> torch.Tensor:
    """Calculate cosine similarity between two sets of embeddings"""
    # Normalize embeddings
    embeddings1 = F.normalize(embeddings1, p=2, dim=1)
    embeddings2 = F.normalize(embeddings2, p=2, dim=1)
    
    # Calculate cosine similarity
    return torch.mm(embeddings1, embeddings2.t())

def calculate_distances_and_track_examples(
    emojis: List[str], 
    emoji_embeddings: torch.Tensor,
    desc_embeddings: torch.Tensor,
    emotion_embeddings: torch.Tensor,
    descriptions: List[str],
    emotions: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate distances and track examples"""
    # Calculate cosine similarities
    emoji_to_desc_sim = calculate_cosine_similarity(emoji_embeddings, desc_embeddings)
    emoji_to_emotion_sim = calculate_cosine_similarity(emoji_embeddings, emotion_embeddings)
    
    # Convert to distances (1 - similarity)
    emoji_to_desc_dist = 1 - emoji_to_desc_sim.diag().cpu().numpy()
    emoji_to_emotion_dist = 1 - emoji_to_emotion_sim.diag().cpu().numpy()
    
    # Create a DataFrame for all results
    results_df = pd.DataFrame({
        "emoji": emojis,
        "description": descriptions,
        "emotion": emotions,
        "desc_distance": emoji_to_desc_dist,
        "emotion_distance": emoji_to_emotion_dist,
        "closer_to": ["description" if d < e else "emotion" for d, e in zip(emoji_to_desc_dist, emoji_to_emotion_dist)]
    })
    
    # Sort by interesting examples (biggest difference in distances)
    results_df["distance_diff"] = abs(results_df["desc_distance"] - results_df["emotion_distance"])
    results_df = results_df.sort_values("distance_diff", ascending=False)
    
    # Select top examples for detailed tracking
    examples_df = results_df.head(10)
    
    return results_df, examples_df

def visualize_embeddings_3d(
    emoji_embeddings: torch.Tensor,
    desc_embeddings: torch.Tensor,
    emotion_embeddings: torch.Tensor,
    emojis: List[str]
) -> None:
    """Create a 3D visualization of the embeddings"""
    # Combine all embeddings for dimensionality reduction
    all_embeddings = torch.cat([emoji_embeddings, desc_embeddings, emotion_embeddings], dim=0).cpu().numpy()
    
    # Use PCA to reduce to 3 dimensions
    pca = PCA(n_components=3)
    embeddings_3d = pca.fit_transform(all_embeddings)
    
    # Split back into separate sets
    n = len(emojis)
    emoji_3d = embeddings_3d[:n]
    desc_3d = embeddings_3d[n:2*n]
    emotion_3d = embeddings_3d[2*n:3*n]
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points
    ax.scatter(emoji_3d[:, 0], emoji_3d[:, 1], emoji_3d[:, 2], c='blue', label='Emojis', alpha=0.8)
    ax.scatter(desc_3d[:, 0], desc_3d[:, 1], desc_3d[:, 2], c='green', label='Descriptions', alpha=0.5)
    ax.scatter(emotion_3d[:, 0], emotion_3d[:, 1], emotion_3d[:, 2], c='red', label='Emotions', alpha=0.5)
    
    # Connect emojis to their descriptions and emotions with lines
    for i in range(min(20, n)):  # Limit to 20 examples to avoid clutter
        ax.plot(
            [emoji_3d[i, 0], desc_3d[i, 0]],
            [emoji_3d[i, 1], desc_3d[i, 1]],
            [emoji_3d[i, 2], desc_3d[i, 2]],
            'g--', alpha=0.3
        )
        ax.plot(
            [emoji_3d[i, 0], emotion_3d[i, 0]],
            [emoji_3d[i, 1], emotion_3d[i, 1]],
            [emoji_3d[i, 2], emotion_3d[i, 2]],
            'r--', alpha=0.3
        )
        
        # Add emoji labels for a few examples
        ax.text(emoji_3d[i, 0], emoji_3d[i, 1], emoji_3d[i, 2], emojis[i], fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')
    ax.set_zlabel('PCA Component 3')
    ax.set_title('3D Visualization of Emoji, Description, and Emotion Embeddings')
    ax.legend()
    
    # Save the figure
    plt.savefig("emoji_embeddings_3d.png", dpi=300, bbox_inches='tight')
    print("3D visualization saved as emoji_embeddings_3d.png")
    
    # Try to show the plot
    try:
        plt.show()
    except:
        print("Unable to display the plot. The figure has been saved to disk.")

def create_alternative_visualization(
    results_df: pd.DataFrame,
    emoji_embeddings: torch.Tensor
) -> None:
    """Create an alternative 2D visualization of emoji embeddings, colored by closer_to"""
    # Use t-SNE for better 2D visualization of emoji embeddings
    tsne = TSNE(n_components=2, random_state=42)
    emoji_2d = tsne.fit_transform(emoji_embeddings.cpu().numpy())
    
    # Create a new figure
    plt.figure(figsize=(12, 10))
    
    # Get colors based on whether emoji is closer to description or emotion
    colors = ['g' if closer == 'description' else 'r' for closer in results_df['closer_to']]
    
    # Create scatter plot
    scatter = plt.scatter(emoji_2d[:, 0], emoji_2d[:, 1], c=colors, alpha=0.7)
    
    # Add emoji labels for some examples
    for i in range(min(30, len(results_df))):
        plt.annotate(
            results_df.iloc[i]['emoji'], 
            (emoji_2d[i, 0], emoji_2d[i, 1]),
            fontsize=12
        )
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=10, label='Closer to Description'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=10, label='Closer to Emotion')
    ]
    plt.legend(handles=legend_elements)
    
    # Set title and labels
    plt.title('2D t-SNE Visualization of Emoji Embeddings')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    
    # Save the figure
    plt.savefig("emoji_embeddings_2d.png", dpi=300, bbox_inches='tight')
    print("2D visualization saved as emoji_embeddings_2d.png")
    
    # Try to show the plot
    try:
        plt.show()
    except:
        print("Unable to display the plot. The figure has been saved to disk.")

def main():
    # Load emoji analysis data
    print("Loading emoji analysis data...")
    emoji_data = load_emoji_analysis()
    
    if not emoji_data:
        return
    
    print(f"Loaded {len(emoji_data)} emoji analyses")
    
    # Prepare data for encoding
    print("Preparing data for encoding...")
    emojis, descriptions, emotions, emotion_explanations = prepare_data_for_encoding(emoji_data)
    
    # Initialize encoder and tokenizer
    print(f"Initializing LlamaEncoder with model: {MODEL_NAME}")
    encoder = LlamaEncoder(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set padding token for the tokenizer (required for batch processing)
    if tokenizer.pad_token is None:
        print("Setting up padding token for the tokenizer...")
        tokenizer.pad_token = tokenizer.eos_token
    
    # Encode emojis, descriptions, and emotions
    print("Encoding emojis...")
    emoji_embeddings = encode_texts(emojis, encoder, tokenizer)
    
    print("Encoding descriptions...")
    desc_embeddings = encode_texts(descriptions, encoder, tokenizer)
    
    print("Encoding emotions...")
    emotion_embeddings = encode_texts(emotions, encoder, tokenizer)
    
    # Calculate distances and track examples
    print("Calculating distances and tracking examples...")
    results_df, examples_df = calculate_distances_and_track_examples(
        emojis, emoji_embeddings, desc_embeddings, emotion_embeddings, descriptions, emotions
    )
    
    # Save results to CSV
    results_df.to_csv(RESULTS_FILE, index=False)
    print(f"Results saved to {RESULTS_FILE}")
    
    # Print example results
    print("\nTop Examples:")
    for i, row in examples_df.iterrows():
        print(f"{row['emoji']} - Description: {row['description'][:50]}... (dist: {row['desc_distance']:.4f})")
        print(f"  Emotion: {row['emotion']} (dist: {row['emotion_distance']:.4f})")
        print(f"  Closer to: {row['closer_to']}\n")
    
    # Analyze overall trends
    closer_to_desc = (results_df['closer_to'] == 'description').sum()
    closer_to_emotion = (results_df['closer_to'] == 'emotion').sum()
    print(f"\nOverall: {closer_to_desc} emojis closer to their descriptions, {closer_to_emotion} closer to their emotions")
    
    # Create 3D visualization
    print("\nCreating 3D visualization...")
    visualize_embeddings_3d(emoji_embeddings, desc_embeddings, emotion_embeddings, emojis)
    
    # Create alternative 2D visualization
    print("\nCreating alternative 2D visualization...")
    create_alternative_visualization(results_df, emoji_embeddings)
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
