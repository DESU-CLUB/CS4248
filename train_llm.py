from llm import GPT2Adapter, EmojiLLM
from encoder import Encoder
from decoder import CustomDecoder
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
import datasets
import torch
import torch.nn as nn
import wandb
import os
import math
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR


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
        
        # Store reference to the tokenizer's vocab size for verification
        self.target_vocab_size = decoder.token_embedding.weight.size(0)
        print(f"Target vocabulary size from decoder: {self.target_vocab_size}")

    def forward(self, x):
        # Process through encoder (get embeddings)
        with torch.no_grad():  # Freeze encoder during training
            encoder_outputs = self.encoder(x)
        
        # Process through LLM
        llm_outputs = self.llm(encoder_outputs)
        
        # Debug shape before decoder
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        # Process through decoder
        # Note: The decoder needs both the encoder outputs (llm_outputs in our case)
        # and the target tokens (x) for teacher forcing during training
        try:
            # Call decoder with both llm_outputs and input_ids for teacher forcing
            logits = self.decoder(llm_outputs, x)
            
            # The decoder outputs logits directly, not actual generated tokens
            # These logits are predictions for each token in the sequence
            
            # Check and fix shapes if needed
            if logits.size(0) != batch_size:
                print(f"WARNING: Decoder output batch size {logits.size(0)} doesn't match expected {batch_size}")
            
            # Reshape the logits if needed - but maintain the actual vocabulary dimension
            # because that's what the loss function expects
            return logits
            
        except Exception as e:
            print(f"Error in decoder: {e}")
            # Create dummy output with correct shape as a fallback
            outputs = torch.zeros((batch_size, seq_len, self.target_vocab_size), 
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
    
    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW([
        {'params': model.llm.parameters(), 'weight_decay': 0.01}, 
        {'params': model.decoder.parameters(), 'weight_decay': 0.01}
    ], lr=learning_rate, betas=(0.9, 0.999), eps=1e-8)
    
    # Calculate total training steps for scheduler
    total_steps = len(train_loader) * epochs
    warmup_steps = len(train_loader)  # One epoch of warmup
    
    # Create a learning rate scheduler with warmup
    def lr_lambda(current_step):
        # Linear warmup for warmup_steps, then cosine decay
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine annealing
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Initialize loss function - we'll apply our own masking
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # Initialize wandb
    wandb.init(project="emoji-llm-training")
    wandb.config.update({
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "model": "FullEmojiLLM",
        "warmup_steps": warmup_steps,
        "weight_decay": 0.01,
        "optimizer": "AdamW",
        "scheduler": "Warmup + Cosine"
    })
    
    # Freeze encoder parameters
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    # Training loop
    model.train()
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        total_tokens = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            input_ids = batch["input_ids"].to(device)
            loss_mask = batch["loss_mask"].to(device)
            
            # Debug initial shapes
            print(f"[Batch {batch_idx}] Initial input_ids shape: {input_ids.shape}")
            
            # For decoder, we need target_ids shifted by one position
            # This creates the autoregressive target where we predict the next token
            if input_ids.size(1) > 1:  # Make sure we have enough tokens
                target_ids = input_ids[:, 1:].clone()  # Remove first token (BOS token)
                # Also slice the loss mask to match the target_ids shape
                loss_mask = loss_mask[:, 1:].clone()
            else:
                # Handle edge case with very short sequences
                print("WARNING: Sequence too short for proper slicing")
                target_ids = input_ids.clone()
            
            # Print debug info for first batch of each epoch
            if batch_idx == 0:
                print(f"[Epoch {epoch+1}] After slicing:")
                print(f"  input_ids shape: {input_ids.shape}")
                print(f"  target_ids shape: {target_ids.shape}")
                print(f"  loss_mask shape: {loss_mask.shape}")
                if epoch == 0:
                    print(f"  Vocabulary size from tokenizer: {tokenizer.vocab_size}")
                    # Show a sample of the data
                    print(f"  Sample token IDs (input): {input_ids[0, :10]}")
                    print(f"  Sample token IDs (target): {target_ids[0, :10]}")
            
            # Zero gradients
            optimizer.zero_grad()
            
            try:
                # Forward pass - get logits
                logits = model(input_ids)
                
                # Print shape debugging
                if batch_idx == 0:
                    print(f"[Epoch {epoch+1}] After forward pass:")
                    print(f"  logits shape: {logits.shape}")
                    print(f"  target_ids shape: {target_ids.shape}")
                
                # Check if logits and target shapes match
                if logits.size(1) != target_ids.size(1):
                    print(f"WARNING: Sequence length mismatch - logits: {logits.size(1)}, targets: {target_ids.size(1)}")
                    
                    # Adjust shapes to match
                    min_seq_len = min(logits.size(1), target_ids.size(1))
                    if logits.size(1) > min_seq_len:
                        logits = logits[:, :min_seq_len, :]
                        print(f"  Truncated logits to shape: {logits.shape}")
                    if target_ids.size(1) > min_seq_len:
                        target_ids = target_ids[:, :min_seq_len]
                        loss_mask = loss_mask[:, :min_seq_len]
                        print(f"  Truncated targets to shape: {target_ids.shape}")
                
                # Calculate per-token loss - use reshape instead of view to avoid contiguity issues
                flat_logits = logits.reshape(-1, logits.size(-1))
                flat_targets = target_ids.reshape(-1)
                
                # Print final shapes for loss calculation
                if batch_idx == 0 and epoch == 0:
                    print(f"Loss calculation shapes:")
                    print(f"  flat_logits: {flat_logits.shape}")
                    print(f"  flat_targets: {flat_targets.shape}")
                
                # Manually create a new tensor to ensure contiguity if needed
                if not flat_logits.is_contiguous():
                    flat_logits = flat_logits.contiguous()
                if not flat_targets.is_contiguous():
                    flat_targets = flat_targets.contiguous()
                
                token_loss = criterion(flat_logits, flat_targets)
                
                # Reshape back to match target shape
                token_loss = token_loss.reshape(target_ids.size())
                
                # Apply loss mask
                masked_loss = token_loss * loss_mask.float()
                
                # Average loss over only the masked tokens
                num_masked_tokens = loss_mask.sum().item()
                if num_masked_tokens > 0:
                    loss = masked_loss.sum() / num_masked_tokens
                    
                    # Backward pass
                    loss.backward()
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    # Update parameters
                    optimizer.step()
                    
                    # Update learning rate scheduler
                    scheduler.step()
                    current_lr = scheduler.get_last_lr()[0]
                    
                    # Log batch loss
                    epoch_loss += loss.item() * num_masked_tokens
                    total_tokens += num_masked_tokens
                    global_step += 1
                    
                    if batch_idx % 10 == 0:
                        print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, "
                              f"Loss: {loss.item():.4f}, LR: {current_lr:.8f}, Masked tokens: {num_masked_tokens}")
                        wandb.log({
                            "batch_loss": loss.item(),
                            "masked_tokens": num_masked_tokens,
                            "learning_rate": current_lr,
                            "step": global_step
                        })
                else:
                    print(f"Skipping batch {batch_idx} - no valid tokens for loss calculation")
                    
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {e}")
                if "CUDA out of memory" in str(e):
                    print("CUDA OOM error - trying to free memory")
                    if 'logits' in locals():
                        del logits
                    if 'token_loss' in locals():
                        del token_loss
                    if 'loss' in locals():
                        del loss
                    torch.cuda.empty_cache()
                continue
        
        # Log epoch loss (average per token)
        if total_tokens > 0:
            avg_epoch_loss = epoch_loss / total_tokens
            print(f"Epoch: {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}, Total tokens: {total_tokens}")
            wandb.log({
                "epoch": epoch+1,
                "train_loss": avg_epoch_loss,
                "total_tokens": total_tokens
            })
            
            # Save model if loss improved
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                # Create checkpoints directory if it doesn't exist
                os.makedirs("checkpoints", exist_ok=True)
                try:
                    # Save best model
                    torch.save(model.state_dict(), "checkpoints/best_model.pt", _use_new_zipfile_serialization=False)
                    print(f"Saved new best model with loss: {best_loss:.4f}")
                except Exception as e:
                    print(f"Error saving best model: {e}")
                
        else:
            print(f"Epoch: {epoch+1}/{epochs}, No valid tokens processed")
        
        # No checkpoint saving during training - only save at the end and best model
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    # Save final model with error handling
    try:
        torch.save(model.state_dict(), "checkpoints/trained_full_emoji_llm.pt", _use_new_zipfile_serialization=False)
        print("Successfully saved final model")
    except Exception as e:
        print(f"Error saving final model: {e}")
        try:
            # Try alternative saving approach
            print("Trying alternative saving approach...")
            cpu_model = model.to("cpu")
            torch.save(cpu_model.state_dict(), "checkpoints/trained_full_emoji_llm_cpu.pt", _use_new_zipfile_serialization=False)
            model = model.to(device)  # Move model back to original device
            print("Successfully saved final model using alternative approach")
        except Exception as e2:
            print(f"All saving attempts failed: {e2}")
            
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