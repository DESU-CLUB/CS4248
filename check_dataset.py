# Just run llama encoder on the dataset

import torch
import pandas as pd

#  Run through datasets/elco_adapted_edited.csv and text2emoji.csv with the llama tokenizer

#Use batch decode to go through as many as possible
import os
from transformers import AutoTokenizer
from tqdm import tqdm

def check_dataset_tokenization(file_path, model_name="meta-llama/Llama-3.2-1B-Instruct", batch_size=32):
    """
    Check if all texts in a dataset can be properly tokenized by the Llama tokenizer.
    
    Args:
        file_path: Path to the CSV file
        model_name: Name of the model to use for tokenization
        batch_size: Number of samples to process at once
    
    Returns:
        List of problematic entries that couldn't be properly tokenized
    """
    print(f"Checking tokenization for dataset: {file_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Determine which column contains the text
    text_column = None
    for col in df.columns:
        if 'text' in col.lower() or 'prompt' in col.lower() or 'input' in col.lower():
            text_column = col
            break
    
    # Determine which column contains the emoji
    emoji_column = None
    for col in df.columns:
        if 'emoji' in col.lower() or 'output' in col.lower():
            emoji_column = col
            break
    
    if text_column is None:
        if len(df.columns) >= 1:
            text_column = df.columns[0]
        else:
            raise ValueError(f"Could not determine text column in {file_path}")
    
    print(f"Using column: '{text_column}' for text")
    
    # Process each entry one by one
    problematic_entries = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        text = str(row[text_column])
        
        try:
            # Tokenize the text
            tokens = tokenizer.encode(text)
            
            # Decode back to check if information is preserved
            decoded = tokenizer.decode(tokens)
            
            # Check if the decoded text is significantly different
            if len(decoded) < len(text) * 0.8:  # If we lost more than 20% of the content
                problematic_entries.append({
                    'index': idx,
                    'original': text,
                    'decoded': decoded[:100],  # First 100 chars
                    'error': 'Significant information loss during tokenization'
                })
        except Exception as e:
            problematic_entries.append({
                'index': idx,
                'original': text,
                'error': str(e)
            })
    
    return problematic_entries

# Check both datasets
datasets = [
    "datasets/ELCo_adapted_edited.csv",
    "datasets/text2emoji.csv"
]

for dataset_path in datasets:
    if os.path.exists(dataset_path):
        problems = check_dataset_tokenization(dataset_path)
        
        if problems:
            print(f"\nFound {len(problems)} problematic entries in {dataset_path}:")
            for i, problem in enumerate(problems[:10]):  # Show first 10
                print(f"Entry {problem['index']}: {problem.get('error', 'Tokenization issue')}")
                print(f"  Original: {problem['original']}")
                if 'decoded' in problem:
                    print(f"  Decoded: {problem['decoded']}...")
                print()
            
            if len(problems) > 10:
                print(f"... and {len(problems) - 10} more issues")
        else:
            print(f"\nNo tokenization issues found in {dataset_path}")
    else:
        print(f"\nDataset file not found: {dataset_path}")



