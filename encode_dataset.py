import torch
from transformers import AutoTokenizer
from llama_encoder import LlamaEncoder

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
import pandas as pd
import os
from tqdm import tqdm
import numpy as np

def encode_dataset(file_path, output_path=None, model_name="meta-llama/Llama-3.2-1B-Instruct", batch_size=16, max_length=None):
    """
    Encode emoji in a dataset using the Llama model and save the results.
    
    Args:
        file_path: Path to the CSV file with Text and Emoji columns or a pandas DataFrame
        output_path: Path to save the encoded dataset (defaults to original filename with _encoded suffix)
        model_name: Name of the model to use for encoding
        batch_size: Number of samples to process at once
        max_length: Maximum sequence length for padding (if None, will use the largest sequence)
    
    Returns:
        Path to the saved encoded dataset or the DataFrame if no output_path
    """
    print(f"Encoding emojis in dataset")
    
    # Handle both DataFrame and file path inputs
    if isinstance(file_path, pd.DataFrame):
        df = file_path
        if output_path is None:
            output_path = "combined_dataset_encoded.csv"
    else:
        # Set default output path if not provided
        if output_path is None:
            base_name = os.path.splitext(file_path)[0]
            output_path = f"{base_name}_encoded.csv"
        
        # Load dataset
        print(f"Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)
    
    # Create directory for vector files if it doesn't exist
    vector_dir = "datasets/vector_files"
    os.makedirs(vector_dir, exist_ok=True)
    print(f"Vector files will be stored in: {vector_dir}")
    
    # Determine which columns contain text and emoji
    text_column = None
    emoji_column = None
    
    for col in df.columns:
        if 'text' in col.lower() or 'prompt' in col.lower() or 'input' in col.lower():
            text_column = col
        elif 'emoji' in col.lower() or 'output' in col.lower():
            emoji_column = col
    
    if text_column is None or emoji_column is None:
        if len(df.columns) >= 2:
            text_column = df.columns[0]
            emoji_column = df.columns[1]
        else:
            raise ValueError(f"Could not determine text and emoji columns in {file_path}")
    
    print(f"Using columns: '{text_column}' for text and '{emoji_column}' for emoji")
    print(f"Encoding only the emoji column: '{emoji_column}'")
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = LlamaEncoder(model_name).to(device)
    encoder.eval()
    
    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
    
    # Determine max sequence length if not provided
    if max_length is None:
        # First tokenize all texts to find maximum length
        all_emojis = df[emoji_column]
        # Find if nans in emoji column
        lengths = [len(tokenizer.encode(emoji)) for emoji in all_emojis]
        max_length = max(lengths)
        print(f"Maximum sequence length in dataset: {max_length}")
    else:
        print(f"Using provided maximum length: {max_length}")
    
    # Process in batches
    with torch.no_grad():
        vector_paths = []
        for i in tqdm(range(0, len(df), batch_size)):
            # Get batch of emojis
            batch_emojis = df[emoji_column].iloc[i:i+batch_size].tolist()
            
            # Tokenize batch
            encoded = tokenizer(batch_emojis, return_tensors="pt", padding='max_length', 
                               truncation=True, max_length=max_length)
            input_ids = encoded.input_ids.to(device)
 
            # Get embeddings
            outputs = encoder(input_ids)
            
            # Convert to numpy and store
            batch_embeddings = outputs.cpu().numpy()
            
            # Save each embedding as an npz file
            for j, embedding in enumerate(batch_embeddings):
                idx = i + j
                vector_filename = f"vector_{idx}.npz"
                vector_path = os.path.join(vector_dir, vector_filename)
                np.savez_compressed(vector_path, embedding=embedding)
                vector_paths.append(vector_filename)
    
    # Add column with paths to vector files
    df['vector'] = vector_paths
    
    # Keep the encoded_emoji_vector column for backward compatibility
    encoded_strings = [np.array2string(np.load(os.path.join(vector_dir, path))['embedding'], 
                       separator=',', precision=6)[1:-1] for path in vector_paths]
    df['encoded_emoji_vector'] = encoded_strings
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Encoded dataset saved to: {output_path}")
    print(f"Vector files saved to: {vector_dir}")
    
    return output_path

if __name__ == "__main__":
    # Paths to datasets
    datasets = [
        "datasets/ELCo_adapted_edited.csv",
        "datasets/text2emoji.csv"
    ]
    
    # Combine datasets
    combined_df = pd.DataFrame()
    
    for dataset_path in datasets:
        if os.path.exists(dataset_path):
            print(f"Loading dataset: {dataset_path}")
            df = pd.read_csv(dataset_path)
            # Append to combined dataframe
            combined_df = pd.concat([combined_df, df], ignore_index=True)
        else:
            print(f"Dataset file not found: {dataset_path}")
    
    if not combined_df.empty:
        print(f"Combined dataset size: {len(combined_df)} rows")
        output_path = "datasets/combined_dataset_encoded.csv"
        # Encode the combined dataset
        encode_dataset(combined_df, output_path=output_path)
    else:
        print("No data was loaded. Please check your dataset paths.")
