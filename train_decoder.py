import transformers
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import datasets
from datasets import load_dataset
import decoder
from encoder import Encoder
from transformers import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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

def setup(rank, world_size):
    """Initialize the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

def train_decoder_distributed(rank, world_size, num_epochs=15):
    # Set up the distributed environment
    setup(rank, world_size)
    
    # Set device for this process
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # dataset definition and preprocessing
    ds = load_dataset("KomeijiForce/Text2Emoji")
    X = ds["train"]['text']
    
    # Split the input text for train/test
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=0)
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    X_train = [str(x) for x in X_train]
    X_test = [str(x) for x in X_test]

    if rank == 0:
        print("Tokenizing training data...")
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
    if rank == 0:
        print("Tokenizing test data...")
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, return_tensors="pt")

    voc_size = tokenizer.vocab_size
    embed_size = 128
    num_heads = 8
    num_layers = 6
    epochs = num_epochs
    lr = 1e-3
    batch_size = 8  # Per GPU batch size

    # Use the same X for both input and target (reconstruction task)
    train_dataset = EncoderDataset(X_train_tokenized['input_ids'], 
                                  X_train_tokenized['input_ids'])
    test_dataset = EncoderDataset(X_test_tokenized['input_ids'], 
                                 X_test_tokenized['input_ids'])

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    
    # Initialize the encoder
    encoder = Encoder(voc_size, embed_size, num_heads, num_layers, device).to(device)
    encoder.load_state_dict(state_dict)

    # Create our custom decoder
    decoder_model = decoder.CustomDecoder(
        vocab_size=voc_size,
        d_model=512,
        nhead=8,
        num_decoder_layers=4,
        dim_feedforward=2048, 
        dropout=0.1,
        encoder_dim=2048  # Updated to match the 2048-dim encoder embeddings
    ).to(device)
    
    # Wrap models with DDP
    encoder = DDP(encoder, device_ids=[rank], output_device=rank)
    decoder_model = DDP(decoder_model, device_ids=[rank], output_device=rank)

    optimizer = torch.optim.Adam(decoder_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    # Create data loaders with distributed sampler
    loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    if rank == 0:
        print("Training dataset size: ", len(train_dataset))
        print("Test dataset size: ", len(test_dataset))
        print("Start training...")
        
        # Initialize WandB only on the main process
        wandb.init(project="emoji_encoder_elco")
        wandb.config = {
            "learning_rate": lr,
            "epochs": epochs,
            "batch_size": batch_size * world_size,  # Global batch size
            "embed_size": embed_size,
            "num_heads": num_heads,
            "num_layers": num_layers
        }
        wandb.watch(decoder_model, log="all")

    # Freeze encoder weights
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    for epoch in range(epochs):
        # Set epoch for the sampler
        train_sampler.set_epoch(epoch)
        
        decoder_model.train()
        epoch_loss = 0.0
        
        for batch in loader:
            input_ids, target_ids = batch
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            optimizer.zero_grad()
            
            # Get encoder embeddings
            with torch.no_grad():
                encoder_outputs = encoder(input_ids)
            
            # Forward pass through decoder - reconstruct the original input
            logits = decoder_model(encoder_outputs, target_ids)
            
            # Calculate loss - we want the decoder to reconstruct the original input
            # Shift logits and labels for teacher forcing
            shift_logits = logits
            shift_labels = target_ids[:, 1:]  # Remove BOS token
            
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), 
                            shift_labels.reshape(-1))
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            
            # Calculate local loss
            local_loss = loss.item()
            
            # Gather and average loss across all processes
            all_losses = [torch.zeros(1).to(device) for _ in range(world_size)]
            torch_loss = torch.tensor([local_loss], device=device)
            dist.all_gather(all_losses, torch_loss)
            
            # Sum up the losses
            if rank == 0:
                avg_loss = sum([l.item() for l in all_losses]) / world_size
                epoch_loss += avg_loss
                
                # Log to WandB on main process
                wandb.log({
                    "train_loss": avg_loss,
                    "epoch": epoch,
                    "learning_rate": scheduler.get_last_lr()[0]
                })

        # Update learning rate scheduler
        scheduler.step()
        
        # Synchronize before logging epoch results
        dist.barrier()
        
        # Log epoch results on main process
        if rank == 0:
            total_batches = len(loader)
            wandb.log({
                "epoch_loss": epoch_loss / total_batches,
                "epoch": epoch
            })
            print(f"Epoch {epoch}, Loss: {epoch_loss / total_batches}")

    # Save the trained decoder model on the main process
    if rank == 0:
        torch.save(decoder_model.module.state_dict(), "trained_decoder_model.pt")
        wandb.finish()

    # Clean up distributed environment
    cleanup()

def main():
    # Get the number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Using {world_size} GPUs for training")
    
    # Start multiple processes
    mp.spawn(
        train_decoder_distributed,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()