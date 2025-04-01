import transformers
import torch
import torch.nn as nn
import torch.optim as optim
import datasets
from datasets import load_dataset
import decoder
from encoder import Encoder
from transformers import BertTokenizer, AutoModel, AutoTokenizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import wandb

wandb.init(project="emoji_encoder_elco")
wandb.config = {
    "learning_rate": 1e-3,
    "epochs": 5,
    "batch_size": 8,
    "embed_size": 128,
    "num_heads": 8,
    "num_layers": 6
}
# Load the weights from Hugging Face
state_dict = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/DESUCLUB/emoji_encoder_elco/resolve/main/trained_encoder_model.pt"
)
llm = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")


# Set the model to evaluation mode
decoder = decoder.LlamaWithAdapter(llm, adapter)

# Load the dataset
class EncoderDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)
    

    def __getitem__(self, idx):
        with torch.no_grad():
            return torch.tensor(self.X[idx], dtype=torch.long), self.y[idx]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_decoder(encoder_dataset, num_epochs=10, device=device):
     # dataset definition and preprocessing
    ds = load_dataset("KomeijiForce/Text2Emoji")
    X = ds["train"]['text']
    y = ds['train']['emoji'] #ds["train"]['labels']


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    label_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    if label_tokenizer.pad_token is None:
        label_tokenizer.pad_token = "<|finetune_right_pad_id|>"
    X_train = [str(x) for x in X_train]
    X_test = [str(x) for x in X_test]

    print("Tokenizing training data...")
    X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, return_tensors="pt")
    y_train = label_tokenizer(y_train, padding=True, truncation=True, return_tensors="pt").input_ids
    print("Tokenizing test data...")
    X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, return_tensors="pt")
    y_test = label_tokenizer(y_test, padding=True, truncation=True, return_tensors="pt").input_ids

    voc_size = tokenizer.vocab_size
    embed_size = 128
    num_heads = 8
    num_layers = 6
    epochs = 5
    lr = 1e-3
    batch_size = 8



    train_dataset = EncoderDataset(X_train_tokenized['input_ids'], 
                                y_train)
    test_dataset = EncoderDataset(X_test_tokenized['input_ids'], 
                                y_test)

    encoder = Encoder(voc_size, embed_size, num_heads, num_layers, device).to(device)
    encoder.load_state_dict(state_dict)

    llm = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
    adapter = decoder.Adapter(llm)

    decoder = decoder.LlamaWithAdapter(llm, adapter)

    optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,epochs)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    print("Training dataset size: ", len(train_dataset))
    print("Test dataset size: ", len(test_dataset))
    print("Start training...")
    encoder.freeze_encoder()
    encoder.eval()
    encoder.to(device)

    # At the beginning after model initialization
    wandb.watch(decoder, log="all")  # Track the decoder model

    for epoch in range(epochs):
        decoder.train()
        decoder.to(device)
        epoch_loss = 0
        for batch in loader:
            input_ids, labels = batch
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            
            # Get encoder embeddings
            with torch.no_grad():
                encoder_outputs = encoder(input_ids)
            
            # Forward pass through decoder
            logits = decoder(encoder_outputs)
            
            # Calculate loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Backpropagation
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            # Inside training loop after loss calculation
            wandb.log({
                "train_loss": loss.item(),
                "epoch": epoch,
                "learning_rate": scheduler.get_last_lr()[0]
            })

        # After each epoch
        wandb.log({
            "epoch_loss": epoch_loss/len(loader),
            "epoch": epoch
        })
    wandb.finish()


if __name__ == "__main__":
    train_decoder()