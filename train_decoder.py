import transformers
import torch
import torch.nn as nn
import torch.optim as optim
import datasets
import decoder
from encoder import Encoder
from transformers import BertTokenizer, AutoModel, AutoTokenizer

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

# Initialize the encoder with the correct vocabulary size
voc_size = tokenizer.vocab_size
embed_size = 128
num_heads = 8
num_layers = 6

# Create the encoder model
encoder = Encoder(voc_size, embed_size, num_heads, num_layers)

# Load the weights from Hugging Face
state_dict = torch.hub.load_state_dict_from_url(
    "https://huggingface.co/DESUCLUB/emoji_encoder_elco/resolve/main/trained_encoder_model.pt"
)
encoder.load_state_dict(state_dict)
llm = AutoModel.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
adapter = decoder.Adapter(llm)

# Set the model to evaluation mode
decoder = decoder.LlamaWithAdapter(llm, adapter)