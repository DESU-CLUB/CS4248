import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import decoder
from encoder import Encoder
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import wandb
import os
import random
import numpy as np

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

def train_decoder(num_epochs=15, use_llama_decoder=False, model_name="meta-llama/Llama-3.2-1B-Instruct"):
    """
    Train the decoder model on a single GPU.
    
    Args:
        num_epochs: Number of training epochs
        use_llama_decoder: If True, use LlamaDecoder instead of CustomDecoder
        model_name: Name/path of the Llama model (only used if use_llama_decoder=True)
    """
    # Set device to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Dataset definition and preprocessing
    ds = load_dataset("KomeijiForce/Text2Emoji")
    X = ds["train"]['text']
    
    # Split the input text for train/test
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=0)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    X_train = [str(x) for x in X_train]
    X_test = [str(x) for x in X_test]

    print("Tokenizing training data...")
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
    print("Tokenizing test data...")
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, return_tensors="pt")

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
    
    # Initialize WandB
    wandb.init(project="emoji_decoder_training")
    wandb.config = {
        "learning_rate": lr,
        "epochs": epochs,
        "batch_size": batch_size,
        "embed_size": embed_size,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "decoder_type": "LlamaDecoder" if use_llama_decoder else "CustomDecoder"
    }
    wandb.watch(decoder_model, log="all")

    # Freeze encoder weights
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        decoder_model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(loader):
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            optimizer.zero_grad()
            
            # Get encoder embeddings
            with torch.no_grad():
                encoder_outputs = encoder(input_ids)
            
            # Forward pass through decoder
            if use_llama_decoder:
                # For LlamaDecoder
                logits = decoder_model(encoder_outputs, input_ids=input_ids)
            else:
                # For CustomDecoder
                logits = decoder_model(encoder_outputs, target_ids)
            
            # Calculate loss
            if use_llama_decoder:
                # For LlamaDecoder
                # Shift targets to align with logits (which already exclude first position)
                shift_labels = target_ids[:, 1:]  # Remove first token
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.reshape(-1, logits.size(-1)), 
                                shift_labels.reshape(-1))
            else:
                # For CustomDecoder
                # Shift logits and labels for teacher forcing
                shift_logits = logits
                shift_labels = target_ids[:, 1:]  # Remove first token
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), 
                                shift_labels.reshape(-1))
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Log batch loss
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch: {batch_idx}/{len(loader)}, Loss: {loss.item():.4f}")
                wandb.log({
                    "train_loss": loss.item(),
                    "epoch": epoch,
                    "batch": batch_idx,
                    "learning_rate": scheduler.get_last_lr()[0]
                })

        # Update learning rate scheduler
        scheduler.step()
        
        # Log epoch results
        avg_epoch_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}")
        wandb.log({
            "epoch_loss": avg_epoch_loss,
            "epoch": epoch
        })

    # Save the trained decoder model
    decoder_type = "llama_decoder" if use_llama_decoder else "custom_decoder"
    torch.save(decoder_model.state_dict(), f"trained_{decoder_type}_model.pt")
    wandb.finish()
    
    return decoder_model

def main():
    # Choose decoder type (set to True to use LlamaDecoder)
    use_llama_decoder = True
    
    # Train the decoder
    train_decoder(num_epochs=15, use_llama_decoder=use_llama_decoder)

if __name__ == "__main__":
    main()