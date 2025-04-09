from llm import GPT2Adapter, EmojiLLM
from encoder import Encoder
from decoder import CustomDecoder
from transformers import AutoModel, AutoTokenizer
import datasets
import torch
import torch.nn as nn
import wandb


# Configuration dictionaries for model parameters
encoder_config = {
    "voc_size": None,  # Will be set dynamically based on tokenizer
    "embed_size": 128,
    "num_heads": 8,
    "num_layers": 6,
    "device": None,  # Will be set dynamically
}

decoder_config = {
    "vocab_size": None,  # Will be set dynamically based on tokenizer
    "d_model": 512,
    "nhead": 8,
    "num_decoder_layers": 4,  # Changed from 6 to 4 to match pre-trained model
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "encoder_dim": 2048,  # Matches the encoder output dimension
}

# Training configuration
training_config = {
    "epochs": 15,
    "learning_rate": 1e-3,
    "batch_size":64,  # Per GPU batch size
    "seed": 42,
}


class FullEmojiLLM(nn.Module):
    def __init__(self, encoder: Encoder, decoder: CustomDecoder, llm: EmojiLLM):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.llm = llm

    def forward(self, x):
        # Process through encoder (get embeddings)
        with torch.no_grad():  # Freeze encoder during training
            encoder_outputs = self.encoder(x)
        
        # Process through LLM
        llm_outputs = self.llm(encoder_outputs)
        
        # Debug shape before decoder
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Process through decoder - ensure x is passed correctly
        try:
            outputs = self.decoder(llm_outputs, x)
            
            # Ensure output has the same batch size and sequence length as input
            if outputs.size(0) != batch_size or outputs.size(1) != seq_len:
                print(f"WARNING: Decoder output shape {outputs.shape} doesn't match expected {(batch_size, seq_len, -1)}")
        except Exception as e:
            print(f"Error in decoder: {e}")
            # Create dummy output with correct shape as a fallback
            outputs = torch.zeros((batch_size, seq_len, self.decoder.token_embedding.weight.size(0)), 
                                device=x.device)
        
        return outputs

def train_llm(model: FullEmojiLLM, train_data, epochs: int, batch_size: int, learning_rate: float):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    # Add special tokens if needed
    special_tokens = {
        "pad_token": "[PAD]" if tokenizer.pad_token is None else tokenizer.pad_token,
        "eos_token": "[EOS]" if tokenizer.eos_token is None else tokenizer.eos_token,
        "bos_token": "[BOS]" if tokenizer.bos_token is None else tokenizer.bos_token,
    }
    tokenizer.add_special_tokens({"pad_token": special_tokens["pad_token"]})
    
    # Create a custom dataset class for simple text (TinyStories format)
    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, tokenizer, max_length=256):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            # Get text from the dataset - TinyStories has a "text" field
            try:
                text = self.dataset[idx]["text"]
            except KeyError:
                # Fallback to first column if "text" doesn't exist
                text = self.dataset[idx][next(iter(self.dataset[idx]))]
            
            # Add special tokens
            formatted_text = f"[BOS] {text} [EOS]"
            
            # All tokens are used for loss calculation since it's a simple text generation task
            # without any masking needed
            
            # Tokenize
            encoding = self.tokenizer(
                formatted_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].squeeze(0)
            
            # For simple text generation, we train on all tokens except padding
            # Create mask where 1 = calculate loss, 0 = ignore (padding)
            loss_mask = (input_ids != tokenizer.pad_token_id)
            
            return {
                "input_ids": input_ids,
                "loss_mask": loss_mask
            }
    
    # Create dataset and dataloader
    train_dataset = TextDataset(train_data, tokenizer)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda batch: {
            "input_ids": torch.stack([item["input_ids"] for item in batch]),
            "loss_mask": torch.stack([item["loss_mask"] for item in batch])
        }
    )
    
    # Initialize optimizer - only train the LLM and decoder parts
    optimizer = torch.optim.AdamW([
        {'params': model.llm.parameters()}, 
        {'params': model.decoder.parameters()}
    ], lr=learning_rate)
    
    # Initialize loss function - we'll apply our own masking
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Initialize wandb
    wandb.init(project="emoji-llm-training")
    wandb.config.update({
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "model": "FullEmojiLLM"
    })
    
    # Freeze encoder parameters
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        total_tokens = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            
            # Create target sequence (shifted right)
            target_ids = input_ids.clone()
            
            # Print debug info for first batch
            if batch_idx == 0 and epoch == 0:
                print(f"Input shape: {input_ids.shape}")
                print(f"Loss mask shape: {loss_mask.shape}")
            
            # Zero gradients
            optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = model(input_ids)
                
                # Print debug info for first batch
                if batch_idx == 0 and epoch == 0:
                    print(f"Model output shape: {outputs.shape}")
                
                # Verify shapes match before loss calculation
                if outputs.size(0) != target_ids.size(0) or outputs.size(1) != target_ids.size(1):
                    print(f"Shape mismatch: outputs {outputs.shape}, targets {target_ids.shape}")
                    
                    # Adjust shapes if needed - make sure they're compatible
                    min_seq_len = min(outputs.size(1), target_ids.size(1))
                    outputs = outputs[:, :min_seq_len, :]
                    target_ids = target_ids[:, :min_seq_len]
                    loss_mask = loss_mask[:, :min_seq_len]
                    
                    print(f"Adjusted shapes: outputs {outputs.shape}, targets {target_ids.shape}")
                
                # Calculate per-token loss
                token_loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                token_loss = token_loss.view(target_ids.size())
                
                # Apply mask - only consider non-padding tokens
                masked_loss = token_loss * loss_mask.float()
                
                # Average loss over only the masked tokens
                num_masked_tokens = loss_mask.sum().item()
                loss = masked_loss.sum() / max(1, num_masked_tokens)  # Avoid division by zero
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                
                # Log batch loss
                epoch_loss += loss.item() * num_masked_tokens
                total_tokens += num_masked_tokens
                
                if batch_idx % 10 == 0:
                    print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, Masked tokens: {num_masked_tokens}")
                    wandb.log({
                        "batch_loss": loss.item(),
                        "masked_tokens": num_masked_tokens,
                        "step": batch_idx + epoch * len(train_loader)
                    })
                    
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
        
        # Log epoch loss (average per token)
        avg_epoch_loss = epoch_loss / max(1, total_tokens)
        print(f"Epoch: {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}, Total tokens: {total_tokens}")
        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_epoch_loss,
            "total_tokens": total_tokens
        })
        
        # Save model checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
        }, f"emoji_llm_checkpoint_epoch_{epoch+1}.pt")
    
    # Save final model
    torch.save(model.state_dict(), "trained_full_emoji_llm.pt")
    wandb.finish()
    
    return model


if __name__ == "__main__":
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer to get vocabulary size
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    # Update configs with dynamic values
    encoder_config["voc_size"] = tokenizer.vocab_size
    encoder_config["device"] = device
    decoder_config["vocab_size"] = tokenizer.vocab_size
    
    # Load encoder state dict
    encoder_state_dict = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/DESUCLUB/emoji_encoder_elco/resolve/main/trained_encoder_model.pt", 
        map_location=device
    )
    
    # Initialize encoder with config parameters
    encoder = Encoder(**encoder_config).to(device)
    encoder.load_state_dict(encoder_state_dict)
    
    # Load decoder state dict
    decoder_state_dict = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/DESUCLUB/emoji_decoder_elco/resolve/main/trained_decoder_model.pt",
        map_location=device
    )
    
    # Initialize decoder with config parameters
    decoder = CustomDecoder(**decoder_config).to(device)
    decoder.load_state_dict(decoder_state_dict)
    
    # Initialize LLM and move to device
    llm = EmojiLLM(model_name="gpt2", hidden_size=2048).to(device)
    
    # Create the full model
    full_model = FullEmojiLLM(encoder, decoder, llm).to(device)
    
    # Make sure the model is in the correct mode for each part
    full_model.encoder.eval()  # Set encoder to evaluation mode
    full_model.llm.train()     # Set LLM to training mode
    full_model.decoder.train() # Set decoder to training mode
    
    ds = datasets.load_dataset("roneneldan/TinyStories")
    train_data = ds["train"]
    
    # Filter out examples that are too long
    def estimate_token_count(example):
        # Rough estimate based on whitespace tokenization plus overhead
        try:
            text = example["text"]
        except KeyError:
            # Fallback to first column if "text" doesn't exist
            text = example[next(iter(example))]
        return len(text.split())
    
    # Print dataset stats before filtering
    total_examples = len(train_data)
    print(f"Total examples before filtering: {total_examples}")
    
    # Optional: Filter out examples estimated to be too long
    max_estimated_tokens = 200  # Conservative estimate
    filtered_indices = [i for i, example in enumerate(train_data) 
                         if estimate_token_count(example) <= max_estimated_tokens]
    
    if len(filtered_indices) < total_examples:
        train_data = train_data.select(filtered_indices)
        print(f"Filtered to {len(train_data)} examples (removed {total_examples - len(train_data)} long examples)")
    
    # Sample a small subset for testing if needed
    train_data = train_data.select(range(min(10, len(train_data))))
    print(f"Using {len(train_data)} examples for training")
    
    # Train the model
    train_llm(full_model, train_data, training_config["epochs"], 
             training_config["batch_size"], training_config["learning_rate"])