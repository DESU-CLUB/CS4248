# Separate file for llama encoder
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class LlamaEncoder(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.num_layers = 4  # Number of layers to use
    
    def forward(self, input_ids: torch.Tensor):
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=True
        )
        
        # Get only the first 4 layers (layer 0 is usually the embedding layer)
        # Assuming hidden_states contains the output of each layer
        hidden_states = outputs.hidden_states[:self.num_layers+1]  # +1 if including embedding layer
        
        return hidden_states[-1]
    

if __name__ == "__main__":
    # Test the LlamaEncoder
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    encoder = LlamaEncoder(model_name)
    
    # Example input
    # Create a sentence of emojis
    emoji_text = "🚀 🔥 🤖 🧠 💻 📊 🔍 🌟 ✨ 🎯"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    input_ids = tokenizer(emoji_text, return_tensors="pt").input_ids
    print(len(input_ids[0]))

    # Forward pass
    outputs = encoder(input_ids)
    
    # Print the outputs
    #print(outputs)
    print(outputs.shape)