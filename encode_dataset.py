import torch
from transformers import AutoTokenizer
from llama_encoder import LlamaEncoder
import pandas as pd
import os
from tqdm import tqdm
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import json
import glob


def encode_dataset(
    file_path,
    output_path=None,
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    batch_size=16,
    max_length=None,
    vector_dir=None,
    resume=True,
    hf_ready=True,
    save_every=50000,
):
    """
    Encode emoji in a dataset using the Llama model and save the results using PyArrow.

    Args:
        file_path: Path to the CSV file with Text and Emoji columns or a pandas DataFrame
        output_path: Path to save the encoded dataset (defaults to original filename with _encoded suffix)
        model_name: Name of the model to use for encoding
        batch_size: Number of samples to process at once
        max_length: Maximum sequence length for padding (if None, will use the largest sequence)
        vector_dir: Directory to store vector parquet files
        resume: Whether to resume from the last processed batch
        hf_ready: Whether to create a Hugging Face datasets-compatible format
        save_every: Save vectors to disk only every N examples (to reduce file count)

    Returns:
        Path to the saved encoded dataset metadata
    """
    print(f"Encoding emojis in dataset with PyArrow storage")

    # Handle both DataFrame and file path inputs
    if isinstance(file_path, pd.DataFrame):
        df = file_path
        if output_path is None:
            output_path = "combined_dataset_encoded"
    else:
        # Set default output path if not provided
        if output_path is None:
            base_name = os.path.splitext(file_path)[0]
            output_path = f"{base_name}_encoded"

        # Load dataset
        print(f"Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)

    # Default vector directory if not provided
    if vector_dir is None:
        vector_dir = f"{output_path}_vectors"

    # Create directory for vector files if it doesn't exist
    os.makedirs(vector_dir, exist_ok=True)
    print(f"Vector files will be stored in: {vector_dir}")

    # Metadata file to track progress
    metadata_file = f"{output_path}_metadata.json"

    # Determine which columns contain text and emoji
    text_column = None
    emoji_column = None

    for col in df.columns:
        if "text" in col.lower() or "prompt" in col.lower() or "input" in col.lower():
            text_column = col
        elif "emoji" in col.lower() or "output" in col.lower():
            emoji_column = col

    if text_column is None or emoji_column is None:
        if len(df.columns) >= 2:
            text_column = df.columns[0]
            emoji_column = df.columns[1]
        else:
            raise ValueError(f"Could not determine text and emoji columns")

    print(f"Using columns: '{text_column}' for text and '{emoji_column}' for emoji")
    print(f"Encoding only the emoji column: '{emoji_column}'")

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    encoder = LlamaEncoder(model_name).to(device)
    encoder.eval()

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = "<|finetune_right_pad_id|>"

    # Determine max sequence length if not provided
    if max_length is None:
        # First tokenize all texts to find maximum length
        all_emojis = df[emoji_column].dropna()
        lengths = [len(tokenizer.encode(emoji)) for emoji in all_emojis]
        max_length = max(lengths)
        print(f"Maximum sequence length in dataset: {max_length}")
    else:
        print(f"Using provided maximum length: {max_length}")

    # Check for existing metadata to resume
    last_processed_batch = -1
    total_batches = (len(df) + batch_size - 1) // batch_size

    # Calculate large batch size based on save_every parameter
    large_batch_size = save_every
    large_batch_count = (
        len(df) + large_batch_size - 1
    ) // large_batch_size  # ceiling division

    if resume and os.path.exists(metadata_file):
        try:
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
                last_processed_batch = metadata.get("last_processed_batch", -1)
                print(f"Resuming from batch {last_processed_batch + 1}/{total_batches}")
        except (json.JSONDecodeError, FileNotFoundError):
            print("Couldn't read metadata file. Starting from the beginning.")

    # Create PyArrow schema for vector storage
    # Determine vector dimension from a sample run
    if last_processed_batch == -1:  # Only needed if starting fresh
        with torch.no_grad():
            sample_emoji = df[emoji_column].iloc[0]
            sample_encoded = tokenizer(
                [sample_emoji],
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            sample_input_ids = sample_encoded.input_ids.to(device)
            sample_output = encoder(sample_input_ids)
            vector_dimension = sample_output.shape[1]
            print(f"Vector dimension: {vector_dimension}")

        # Create initial metadata
        metadata = {
            "last_processed_batch": -1,
            "vector_dimension": int(vector_dimension),
            "max_length": max_length,
            "total_samples": len(df),
            "batch_size": batch_size,
            "processing_batch_size": batch_size,
            "save_batch_size": large_batch_size,
            "total_batches": total_batches,
            "total_save_batches": large_batch_count,
            "model_name": model_name,
        }

        # Save initial metadata
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
    else:
        # Load existing metadata
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
            vector_dimension = metadata["vector_dimension"]

    # Process batches
    with torch.no_grad():
        # Storage for accumulating vectors before writing to disk
        accumulated_indices = []
        accumulated_vectors = []
        current_large_batch = -1

        for batch_idx in tqdm(
            range(last_processed_batch + 1, total_batches), desc="Processing batches"
        ):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(df))

            # Determine which large batch this belongs to
            large_batch_idx = start_idx // large_batch_size

            # Get batch of emojis
            batch_df = df.iloc[start_idx:end_idx].copy()
            batch_emojis = batch_df[emoji_column].fillna("").tolist()
            batch_indices = list(range(start_idx, end_idx))

            # Tokenize batch
            encoded = tokenizer(
                batch_emojis,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            input_ids = encoded.input_ids.to(device)

            # Get embeddings
            outputs = encoder(input_ids)
            batch_embeddings = outputs.cpu().numpy()

            # Add to accumulated vectors
            accumulated_indices.extend(batch_indices)

            for vec in batch_embeddings:
                # Convert numpy array to list of lists
                vec_list = vec.tolist()  # This converts the 2D numpy array to a list of lists
                accumulated_vectors.append(vec_list)

            # Check if we've completed a large batch or reached the end
            reached_large_batch_end = (large_batch_idx != current_large_batch) or (
                batch_idx == total_batches - 1
            )

            if reached_large_batch_end and accumulated_vectors:
                # Save the accumulated vectors
                if current_large_batch == -1:
                    current_large_batch = large_batch_idx

                print(
                    f"Writing large batch {current_large_batch} with {len(accumulated_indices)} vectors"
                )

                # Create PyArrow arrays
                indices_array = pa.array(accumulated_indices, pa.int64())
                vectors_array = pa.array(
                    accumulated_vectors,
                    pa.list_(pa.list_(pa.float32()))
                )

                # Build table
                batch_table = pa.Table.from_arrays(
                    [indices_array, vectors_array], names=["index", "vector"]
                )

                # Save batch to parquet
                batch_file = os.path.join(
                    vector_dir, f"large_batch_{current_large_batch:05d}.parquet"
                )
                pq.write_table(batch_table, batch_file, compression="zstd")

                # Reset accumulators
                accumulated_indices = []
                accumulated_vectors = []
                current_large_batch = large_batch_idx

            # Update and save metadata after each processing batch (not large batch)
            metadata["last_processed_batch"] = batch_idx
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

    # Create dataframe with vector file references
    print("Creating final dataframe with vector references...")

    # Save the original dataframe without adding unnecessary vector_index
    df_file = f"{output_path}.csv"
    df.to_csv(df_file, index=False)

    # If requested, create a Hugging Face datasets-compatible format
    if hf_ready:
        print("Creating Hugging Face datasets-compatible format...")

        # Create a final parquet file that contains the original data and vectors together
        hf_dataset_path = f"{output_path}_hf_dataset"
        os.makedirs(hf_dataset_path, exist_ok=True)

        # Process the large batches
        large_batch_files = sorted(
            glob.glob(os.path.join(vector_dir, "large_batch_*.parquet"))
        )

        for batch_idx, batch_file in enumerate(
            tqdm(large_batch_files, desc="Creating HF dataset")
        ):
            # Load vector data
            vector_table = pq.read_table(batch_file)
            vector_df = vector_table.to_pandas()

            # Get indices and vectors
            indices = vector_df["index"].tolist()
            index_to_vector = {
                row["index"]: row["vector"] for _, row in vector_df.iterrows()
            }

            # Get corresponding original data
            batch_df = df.iloc[indices].copy()

            # Add vectors directly to the batch dataframe
            batch_df["embedding"] = [
                index_to_vector.get(i, [0.0] * vector_dimension) for i in indices
            ]

            # Convert to PyArrow table
            batch_table = pa.Table.from_pandas(batch_df)

            # Save as parquet with sharded naming convention
            shard_path = os.path.join(hf_dataset_path, f"part-{batch_idx:05d}.parquet")
            pq.write_table(batch_table, shard_path, compression="zstd")

        # Create dataset_info.json file for HF datasets
        dataset_info = {
            "description": f"Emoji embeddings generated using {model_name}",
            "citation": "",
            "homepage": "",
            "license": "",
            "features": {
                text_column: {"dtype": "string", "_type": "Value"},
                emoji_column: {"dtype": "string", "_type": "Value"},
                "embedding": {
                    "feature": {
                        "feature": {"dtype": "float32", "_type": "Value"},
                        "_type": "Sequence",
                    },
                    "_type": "Sequence",
                },
            },
            "supervised_keys": None,
            "task_templates": [],
            "builder_name": "parquet",
        }

        with open(os.path.join(hf_dataset_path, "dataset_info.json"), "w") as f:
            json.dump(dataset_info, f, indent=2)

        print(f"Hugging Face dataset created at: {hf_dataset_path}")
        print(
            "You can now upload this folder to the Hugging Face Hub or load it locally with:"
        )
        print(
            f"from datasets import load_from_disk\ndataset = load_from_disk('{hf_dataset_path}')"
        )

    print(f"Encoded dataset metadata saved to: {metadata_file}")
    print(f"Original data with vector indices saved to: {df_file}")
    print(f"Vector files saved to: {vector_dir}")
    print(f"Processing completed: {total_batches} batches")

    return metadata_file


def load_vectors(dataset_path, indices=None):
    """
    Load vectors from PyArrow files for specified indices.

    Args:
        dataset_path: Path to the encoded dataset (without .csv extension)
        indices: List of indices to load (if None, loads all)

    Returns:
        Dictionary of {index: vector} or numpy array of vectors if indices provided
    """
    metadata_file = f"{dataset_path}_metadata.json"
    vector_dir = f"{dataset_path}_vectors"

    # Load metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    save_batch_size = metadata.get("save_batch_size", metadata.get("batch_size"))

    if indices is None:
        # Load all vectors
        all_vectors = {}

        # Get all batch files
        batch_files = sorted(
            glob.glob(os.path.join(vector_dir, "large_batch_*.parquet"))
        )

        for batch_file in tqdm(batch_files, desc="Loading all vectors"):
            table = pq.read_table(batch_file)
            df = table.to_pandas()

            for _, row in df.iterrows():
                all_vectors[row["index"]] = np.array(row["vector"])

        return all_vectors
    else:
        # Load specific indices
        result = {}

        # Group indices by large batch for efficient loading
        batch_indices = {}
        for idx in indices:
            batch_idx = idx // save_batch_size
            if batch_idx not in batch_indices:
                batch_indices[batch_idx] = []
            batch_indices[batch_idx].append(idx)

        # Load each required batch
        for batch_idx, batch_indices_list in batch_indices.items():
            batch_file = os.path.join(
                vector_dir, f"large_batch_{batch_idx:05d}.parquet"
            )

            if os.path.exists(batch_file):
                table = pq.read_table(batch_file)
                df = table.to_pandas()

                # Filter for required indices
                for idx in batch_indices_list:
                    row = df[df["index"] == idx]
                    if not row.empty:
                        result[idx] = np.array(row.iloc[0]["vector"])

        # If result should be in original order, convert to numpy array
        if len(result) == len(indices):
            ordered_result = np.array([result[idx] for idx in indices])
            return ordered_result

        return result


def upload_to_hub(dataset_path, hub_dataset_id, private=True):
    """
    Upload the dataset to the Hugging Face Hub.

    Args:
        dataset_path: Path to the Hugging Face-ready dataset
        hub_dataset_id: ID for the dataset on the Hub (e.g., 'username/dataset-name')
        private: Whether the dataset should be private
    """
    try:
        from datasets import load_from_disk
        from huggingface_hub import HfApi

        print(f"Loading dataset from {dataset_path}...")
        dataset = load_from_disk(dataset_path)

        print(f"Pushing dataset to the Hugging Face Hub as {hub_dataset_id}...")
        dataset.push_to_hub(hub_dataset_id, private=private)

        print(
            f"Dataset successfully uploaded to https://huggingface.co/datasets/{hub_dataset_id}"
        )

    except ImportError:
        print(
            "Please install the required packages: pip install datasets huggingface_hub"
        )

    except Exception as e:
        print(f"Error uploading to Hub: {str(e)}")
        print("You may need to login first with `huggingface-cli login`")


if __name__ == "__main__":
    # Paths to datasets
    datasets = ["datasets/ELCo_adapted_edited.csv", "datasets/text2emoji.csv"]

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
        output_path = "datasets/combined_dataset"

        # Encode the combined dataset with resumable processing and HF-ready format
        encode_dataset(
            combined_df,
            output_path=output_path,
            resume=True,
            hf_ready=True,
            save_every=50000,  # Save every 50k examples as requested
        )

        # Uncomment to upload directly to Hugging Face Hub
        # upload_to_hub(
        #     f"{output_path}_hf_dataset",
        #     "your-username/emoji-embeddings",
        #     private=True
        # )

        # Example of how to load the HF dataset locally
        print("\nTo load the dataset locally:")
        print("from datasets import load_from_disk")
        print(f"dataset = load_from_disk('{output_path}_hf_dataset')")
        print("\nTo use the dataset in a model:")
        print("vectors = dataset['embedding']")
    else:
        print("No data was loaded. Please check your dataset paths.")

