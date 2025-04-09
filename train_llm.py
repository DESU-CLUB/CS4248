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
        
        # Process through decoder
        outputs = self.decoder(llm_outputs, x)
        
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
    
    # Create a custom dataset class for the conversation format
    class ConversationDataset(torch.utils.data.Dataset):
        def __init__(self, dataset, tokenizer, max_length=512):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            example = self.dataset[idx]
            system_prompt = example["system"]
            conversations = example["conversations"]
            
            # Format the conversation as a single text with special tokens
            formatted_text = f"[BOS] {system_prompt} "
            
            # Track positions of user vs assistant content for loss masking
            is_assistant_token = []  # 1 for assistant tokens, 0 for system/user tokens
            
            # Initial tokens are system prompt (not assistant)
            system_tokens = len(self.tokenizer.encode(formatted_text)) - 1  # -1 for BOS
            is_assistant_token.extend([0] * system_tokens)
            
            for turn in conversations:
                role = turn["from"]
                content = turn["value"]
                
                if role.lower() == "user":
                    formatted_text += f"[USER] {content} "
                    # Add user tokens to the mask (not used for loss)
                    user_tokens = len(self.tokenizer.encode(f"[USER] {content} ")) - 1  # -1 for special token
                    is_assistant_token.extend([0] * user_tokens)
                else:  # assistant
                    formatted_text += f"[ASSISTANT] {content} "
                    # Add assistant tokens to the mask (used for loss)
                    assistant_tokens = len(self.tokenizer.encode(f"[ASSISTANT] {content} ")) - 1
                    is_assistant_token.extend([1] * assistant_tokens)
            
            # Add EOS token
            formatted_text += "[EOS]"
            is_assistant_token.append(0)  # EOS token is not used for loss
            
            # Tokenize
            encoding = self.tokenizer(
                formatted_text,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            input_ids = encoding["input_ids"].squeeze(0)
            
            # Create loss mask (1 for tokens we want to calculate loss on - assistant tokens)
            # Pad to match the sequence length
            loss_mask = torch.tensor(is_assistant_token[:self.max_length] + 
                                    [0] * max(0, self.max_length - len(is_assistant_token)),
                                    dtype=torch.bool)
            
            return {
                "input_ids": input_ids,
                "loss_mask": loss_mask
            }
    
    # Create dataset and dataloader
    train_dataset = ConversationDataset(train_data, tokenizer)
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
            
            # Zero gradients
            optimizer.zero_grad()
            
            try:
                # Forward pass
                outputs = model(input_ids)
                
                # Calculate loss with masking
                # outputs shape: [batch_size, seq_len, vocab_size]
                # target_ids shape: [batch_size, seq_len]
                # loss_mask shape: [batch_size, seq_len]
                
                # Calculate per-token loss
                token_loss = criterion(outputs.view(-1, outputs.size(-1)), target_ids.view(-1))
                token_loss = token_loss.view(target_ids.size())
                
                # Apply mask - only consider loss for assistant tokens
                masked_loss = token_loss * loss_mask.float()
                
                # Average loss over only the masked tokens
                num_assistant_tokens = loss_mask.sum().item()
                loss = masked_loss.sum() / max(1, num_assistant_tokens)  # Avoid division by zero
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Update parameters
                optimizer.step()
                
                # Log batch loss
                epoch_loss += loss.item() * num_assistant_tokens
                total_tokens += num_assistant_tokens
                
                if batch_idx % 10 == 0:
                    print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, "
                          f"Loss: {loss.item():.4f}, Assistant tokens: {num_assistant_tokens}")
                    wandb.log({
                        "batch_loss": loss.item(),
                        "assistant_tokens": num_assistant_tokens,
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
    
    # Load tokenizer to get vocabulary size
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    # Update configs with dynamic values
    encoder_config["voc_size"] = tokenizer.vocab_size
    encoder_config["device"] = device
    decoder_config["vocab_size"] = tokenizer.vocab_size
    
    # Load encoder state dict
    encoder_state_dict = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/DESUCLUB/emoji_encoder_elco/resolve/main/trained_encoder_model.pt"
    )
    
    # Initialize encoder with config parameters
    encoder = Encoder(**encoder_config).to(device)
    encoder.load_state_dict(encoder_state_dict)
    
    # Load decoder state dict
    decoder_state_dict = torch.hub.load_state_dict_from_url(
        "https://huggingface.co/DESUCLUB/emoji_decoder_elco/resolve/main/trained_decoder_model.pt"
    )
    
    # Initialize decoder with config parameters
    decoder = CustomDecoder(**decoder_config).to(device)
    decoder.load_state_dict(decoder_state_dict)
    
    # Initialize LLM
    llm = EmojiLLM(model_name="gpt2", hidden_size=2048)
    
    # Create the full model
    full_model = FullEmojiLLM(encoder, decoder, llm)

    ds = datasets.load_dataset("bespokelabs/Bespoke-Stratos-17k")
    train_data = ds["train"]["text"]    
    # Train the model
    train_llm(full_model, train_data,  training_config["epochs"], training_config["batch_size"], training_config["learning_rate"])