import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import decoder
from encoder import Encoder
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import os
import random
import numpy as np
from torch.amp import autocast
from dotenv import load_dotenv
from transformers.tokenization_utils_base import BatchEncoding

# Add BatchEncoding to safe globals for torch.load
torch.serialization.add_safe_globals([BatchEncoding])

# Load environment variables
load_dotenv()

# Load the weights from Hugging Face
state_dict = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/DESUCLUB/emoji_encoder_elco/resolve/main/trained_encoder_model.pt"
)

# Load the dataset
class EncoderDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        with torch.no_grad():
            # Use clone() and detach() to properly handle tensors
            return self.X[idx].clone().detach(), self.y[idx].clone().detach()

def set_seed(seed):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_decoder(num_epochs=15, use_llama_decoder=True, model_name="meta-llama/Llama-3.2-1B-Instruct", 
                  use_wandb=True, debug_samples=None, accumulation_steps=4):
    """
    Train the decoder model on a single GPU with advanced optimizations.
    
    Args:
        num_epochs: Number of training epochs
        use_llama_decoder: If True, use LlamaDecoder instead of CustomDecoder
        model_name: Name/path of the Llama model (only used if use_llama_decoder=True)
        use_wandb: Whether to use wandb for logging
        debug_samples: Limit samples for debugging (None for full dataset)
        accumulation_steps: Number of steps to accumulate gradients
    """
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Dataset definition and preprocessing
    ds = load_dataset("DESUCLUB/combined_emoji_data")
    X = ds["train"]['text']
    
    # Limit samples if debugging
    if debug_samples is not None:
        print(f"DEBUG MODE: Using only {debug_samples} samples")
        X = X[:debug_samples]
    
    # Split the input text for train/test
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=0)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    X_train = [str(x) for x in X_train]
    X_test = [str(x) for x in X_test]

    # Create cache directory if it doesn't exist
    os.makedirs("cache", exist_ok=True)
    
    # Define cache file paths
    train_cache_path = f"cache/train_tokenized_{debug_samples if debug_samples else 'full'}.pt"
    test_cache_path = f"cache/test_tokenized_{debug_samples if debug_samples else 'full'}.pt"
    
    # Check if cached tokenized data exists
    if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
        print("Loading tokenized data from cache...")
        try:
            # Try loading with weights_only=False for backward compatibility
            X_train_tokenized = torch.load(train_cache_path, weights_only=False)
            X_test_tokenized = torch.load(test_cache_path, weights_only=False)
            print("Tokenized data loaded from cache successfully")
        except Exception as e:
            print(f"Error loading from cache: {e}")
            print("Falling back to tokenizing data...")
            # Tokenize data if cache loading fails
            print("Tokenizing training data...")
            X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
            print("Tokenizing test data...")
            X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, return_tensors="pt")
            
            # Save tokenized data to cache
            print("Saving tokenized data to cache...")
            torch.save(X_train_tokenized, train_cache_path)
            torch.save(X_test_tokenized, test_cache_path)
            print("Tokenized data saved to cache successfully")
    else:
        print("Tokenizing training data...")
        X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
        print("Tokenizing test data...")
        X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, return_tensors="pt")
        
        # Save tokenized data to cache
        print("Saving tokenized data to cache...")
        torch.save(X_train_tokenized, train_cache_path)
        torch.save(X_test_tokenized, test_cache_path)
        print("Tokenized data saved to cache successfully")

    voc_size = tokenizer.vocab_size
    embed_size = 128
    num_heads = 8
    num_layers = 6
    epochs = num_epochs
    lr = 1e-3
    batch_size = 32  # Increased for single GPU

    # Use the same X for both input and target (reconstruction task)
    train_dataset = EncoderDataset(X_train_tokenized['input_ids'], 
                                  X_train_tokenized['input_ids'])
    test_dataset = EncoderDataset(X_test_tokenized['input_ids'], 
                                 X_test_tokenized['input_ids'])
    
    # Initialize the encoder
    encoder = Encoder(voc_size, embed_size, num_heads, num_layers, device).to(device)
    encoder.load_state_dict(state_dict)

    # Create decoder model - either CustomDecoder or LlamaDecoder
    if use_llama_decoder:
        print(f"Using LlamaDecoder with model: {model_name}")
        decoder_model = decoder.LlamaDecoder(
            model_name=model_name,
            encoder_dim=2048,  # Encoder output dimension
            device=device
        )
    else:
        print("Using CustomDecoder")
        decoder_model = decoder.CustomDecoder(
            vocab_size=voc_size,
            d_model=512,
            nhead=8,
            num_decoder_layers=4,
            dim_feedforward=2048, 
            dropout=0.1,
            encoder_dim=2048  # Encoder output dimension
        ).to(device)

    # Setup optimizer and scheduler
    optimizer = torch.optim.Adam(decoder_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Create data loader
    loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    print("Training dataset size:", len(train_dataset))
    print("Test dataset size:", len(test_dataset))
    print("Start training...")
    
    # Initialize WandB if available and enabled
    if use_wandb:
        try:
            import wandb
            wandb.init(project="emoji_decoder_training")
            wandb.config = {
                "learning_rate": lr,
                "epochs": epochs,
                "batch_size": batch_size,
                "embed_size": embed_size,
                "num_heads": num_heads,
                "num_layers": num_layers,
                "decoder_type": "LlamaDecoder" if use_llama_decoder else "CustomDecoder",
                "accumulation_steps": accumulation_steps,
                "mixed_precision": "bfloat16"
            }
            wandb.watch(decoder_model, log="all")
            print("WandB initialized successfully")
        except Exception as e:
            print(f"Error initializing WandB: {e}")
            use_wandb = False

    # Freeze encoder weights
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    # Configure precision
    try:
        if device.type == 'cuda':
            print("Using BF16 precision for training")
    except Exception as e:
        print(f"Error configuring precision: {e}")

    for epoch in range(epochs):
        decoder_model.train()
        epoch_loss = 0.0
        
        # Clear cache at the start of each epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        
        for i, batch in enumerate(loader):
            # Periodically clear cache to avoid memory fragmentation
            if i % 50 == 0 and device.type == 'cuda':
                torch.cuda.empty_cache()
                
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = input_ids.to(device)
            #print(f"Input IDs shape: {input_ids.shape}, Target IDs shape: {target_ids.shape}") 
            
            # Only zero gradients when starting a new accumulation cycle
            if i % accumulation_steps == 0:
                optimizer.zero_grad()
            
            # Get encoder embeddings
            with torch.no_grad():
                encoder_outputs = encoder(input_ids)
            
            # Use autocast for mixed precision training with bfloat16
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # Forward pass through decoder
                if use_llama_decoder:
                    # For LlamaDecoder - pass both encoder outputs and shifted inputs
                    input_ids_shifted = target_ids[:, :-1]  # Remove last token (for input)
                    shift_labels = target_ids[:, 1:]        # Remove first token (for target)
                    logits = decoder_model(encoder_outputs, input_ids_shifted)
                else:
                    # For CustomDecoder
                    logits = decoder_model(encoder_outputs, target_ids)
                    # Shift logits and labels for teacher forcing
                    shift_logits = logits
                    shift_labels = target_ids[:, 1:]  # Remove first token
                
                # Calculate loss
                loss_fct = nn.CrossEntropyLoss()
                
                if use_llama_decoder:
                    #print(f"LlamaDecoder - logits shape: {logits.shape}, shift_labels shape: {shift_labels.shape}")
                    # Reshape logits to match the vocabulary size expected by CrossEntropyLoss
                    logits = logits.reshape(-1, logits.size(-1)).contiguous()
                    shift_labels = shift_labels.reshape(-1).contiguous()
                    #print(f"LlamaDecoder - reshaped logits: {logits.shape}, reshaped labels: {shift_labels.shape}")
                    loss = loss_fct(logits, shift_labels)
                else:
                    #print(f"CustomDecoder - shift_logits shape: {shift_logits.shape}, shift_labels shape: {shift_labels.shape}")
                    #print(f"CustomDecoder - reshaped logits: {shift_logits.reshape(-1, shift_logits.size(-1)).shape}, reshaped labels: {shift_labels.reshape(-1).shape}")
                    loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), 
                                   shift_labels.reshape(-1))
                
                # Normalize loss for gradient accumulation
                loss = loss / accumulation_steps
            
            # Regular backward pass without scaler
            loss.backward()
            
            # Only update weights after accumulation_steps
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(loader):
                optimizer.step()
            
            # Calculate loss
            local_loss = loss.item() * accumulation_steps  # Rescale for reporting
            epoch_loss += local_loss
            
            # Log periodically
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch: {i}/{len(loader)}, Loss: {local_loss:.4f}")
                # Log to WandB if enabled
                if use_wandb:
                    wandb.log({
                        "train_loss": local_loss,
                        "epoch": epoch,
                        "batch": i,
                        "learning_rate": scheduler.get_last_lr()[0]
                    })

        # Update learning rate scheduler
        scheduler.step()
        
        # Log epoch results
        avg_epoch_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.4f}")
        
        # Log to WandB if enabled
        if use_wandb:
            wandb.log({
                "epoch_loss": avg_epoch_loss,
                "epoch": epoch
            })

    # Save the trained decoder model
    decoder_type = "llama_decoder" if use_llama_decoder else "custom_decoder"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(decoder_model.state_dict(), f"checkpoints/trained_{decoder_type}_model.pt")
    print(f"Model saved to checkpoints/trained_{decoder_type}_model.pt")
    
    # Finalize WandB if enabled
    if use_wandb:
        wandb.finish()
    
    return decoder_model

def main():
    # Check if wandb is available
    use_wandb = True
    try:
        import wandb
        # API key is loaded from .env file by load_dotenv()
        print("wandb is available, will use it for logging")
    except ImportError:
        print("wandb not installed, continuing without logging")
        use_wandb = False
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Running on CPU.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Define training parameters
    num_epochs = 2
    debug_samples = None# Limit to None samples for faster debugging, set to None for full dataset
    use_llama_decoder = True
    accumulation_steps = 4  # Number of batches to accumulate gradients
    
    # Start training
    train_decoder(
        num_epochs=num_epochs, 
        use_llama_decoder=use_llama_decoder, 
        use_wandb=use_wandb, 
        debug_samples=debug_samples,
        accumulation_steps=accumulation_steps
    )

if __name__ == "__main__":
    main()
