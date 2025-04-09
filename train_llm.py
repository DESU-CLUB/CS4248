from llm import GPT2Adapter, EmojiLLM
from encoder import EmojiEncoder, Encoder
from decoder import EmojiDecoder, CustomDecoder
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
    "num_decoder_layers": 6,
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


class FullEmojiLLM:
    def __init__(self, encoder: EmojiEncoder, decoder: EmojiDecoder, llm: EmojiLLM):
        self.encoder = encoder
        self.decoder = decoder
        self.llm = llm

    def forward(self, x):
        x = self.encoder(x)
        x = self.llm(x)
        x = self.decoder(x)
        return x
    
    
def train_llm(model: FullEmojiLLM, train_data: list[str], epochs: int, batch_size: int, learning_rate: float):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Tokenize the data
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    tokenized_data = tokenizer(train_data, padding=True, truncation=True, return_tensors="pt")
    
    # Create dataset and dataloader
    train_dataset = torch.utils.data.TensorDataset(tokenized_data['input_ids'], tokenized_data['input_ids'])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Initialize wandb
    wandb.init(project="emoji-llm-training")
    wandb.config.update({
        "learning_rate": learning_rate,
        "epochs": epochs,
        "batch_size": batch_size,
        "model": "FullEmojiLLM"
    })
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Move data to device
            data = data.to(device)
            targets = targets.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(data)
            
            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            # Log batch loss
            epoch_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch: {epoch+1}/{epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                wandb.log({"batch_loss": loss.item()})
        
        # Log epoch loss
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch: {epoch+1}/{epochs}, Average Loss: {avg_epoch_loss:.4f}")
        wandb.log({
            "epoch": epoch+1,
            "train_loss": avg_epoch_loss
        })
    
    # Save model
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