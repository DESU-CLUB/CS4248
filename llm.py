from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT2Adapter(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.linear = nn.Linear(hidden_size, 4096)
        self.relu = nn.GELU()
        self.linear2 = nn.Linear(4096, 768)

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x

class EmojiLLM(nn.Module):
    def __init__(self, model_name="gpt2", hidden_size=2048):
        super().__init__()
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        self.adapter = GPT2Adapter(hidden_size=hidden_size)
        # Add a non-linear activation function
        self.activation = nn.GELU()  # GELU is commonly used in transformer models
        # Add a projection layer to map GPT-2 output to your desired dimension
        self.projection = nn.Linear(768, hidden_size)
    
    def forward(self, x):
        # Pass through adapter
        x = self.adapter(x)
        # Get GPT-2 output
        gpt2_output = self.gpt2(inputs_embeds=x)
        # Apply activation function to the last hidden state
        activated = self.activation(gpt2_output.last_hidden_state)
        # Apply projection
        projected = self.projection(activated)
        return projected

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")

    # Get the hidden size dimension of GPT-2
    hidden_size = model.config.hidden_size
    print(f"GPT-2 hidden size dimension: {hidden_size}")

    # You can also access other model dimensions
    print(f"Number of attention heads: {model.config.num_attention_heads}")
    print(f"Number of layers: {model.config.num_hidden_layers}")
    print(f"Vocabulary size: {model.config.vocab_size}")

    adapter = GPT2Adapter(hidden_size=2048).to(device)

    #Create a torch.randn tensor of shape (1, 2048)
    x = torch.randn(1, 2048).to(device)

    #Pass the tensor through the adapter
    x = adapter(x)

    #Pass it through the model
    with torch.no_grad():
        output = model(inputs_embeds=x, output_hidden_states=True)
        print("Output shape:", output.last_hidden_state.shape)
        # Print shapes to compare

# This code only runs if the file is executed directly
if __name__ == "__main__":
    main()
  