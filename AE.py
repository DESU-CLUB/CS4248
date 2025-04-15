import torch
from transformers import BertTokenizer, BertModel, GPT2LMHeadModel, GPT2Tokenizer
torch.manual_seed(0) # for reproducibility
torch.set_default_dtype(torch.float32)
from datasets import load_dataset
from huggingface_hub import login
# put huggingface token 

# use GPU if available
if torch.cuda.is_available():
   device = torch.device("cuda")
   print('GPU')
elif torch.backends.mps.is_available(): # Apple GPU
   device = torch.device("mps")
   print('MPS')
else:
   device = torch.device("cpu")
   print("CPU")

class Autoencoder():
   def __init__(self, device):
      super().__init__()
      self.device = device
      self.encoder = BertModel.from_pretrained('bert-base-multilingual-cased')
      self.encoder_state_dict = torch.load('tuned_bert_encoder_model.pt', map_location="cpu")
      self.encoder.load_state_dict(self.encoder_state_dict)
      self.encoder.to(device)
      self.encoder_tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
      self.decoder = GPT2LMHeadModel.from_pretrained('gpt2')
      self.decoder_state_dict = torch.load("tuned_gpt2_decoder_model2.pt", map_location="cpu")
      self.decoder.load_state_dict(self.decoder_state_dict)
      self.decoder.to(device)
      self.decoder_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
   
   def forward(self, x, max_length = 100, temperature = 1.0, top_k = 100):
      x_tokenized = self.encoder_tokenizer(x, padding=True, truncation=True, return_tensors="pt")
      with torch.no_grad():
            input_embeddings = self.encoder(input_ids = x_tokenized['input_ids'].to(self.device), attention_mask = x_tokenized['attention_mask'].to(self.device)).last_hidden_state
      # Use the generate method to produce tokens
      generated_tokens = self.generate(input_embeddings, max_length=max_length, temperature=temperature, top_k=top_k)

      # Decode the generated tokens into text
      decoded_text = [self.decoder_tokenizer.decode(ids, skip_special_tokens=True) for ids in generated_tokens]
      return decoded_text
   
   def __call__(self, x):
      return self.forward(x)
   
   def generate(self, memory, max_length=50, temperature=1.0, top_k=50):
      """Generate text autoregressively using encoder memory."""
      batch_size = memory.size(0)
      device = memory.device

      # Start with an empty sequence (BOS token or zeros)
      input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)

      # Generation loop
      for _ in range(max_length):
         # Decode using GPT-2
         outputs = self.decoder(input_ids=input_ids, attention_mask=None)

         # Get logits for the last token
         next_token_logits = outputs.logits[:, -1, :] / temperature

         # Apply top-k sampling
         if top_k > 0:
               indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
               next_token_logits[indices_to_remove] = float('-inf')

         # Sample the next token
         probs = torch.softmax(next_token_logits, dim=-1)
         next_token = torch.multinomial(probs, num_samples=1)

         # Append the generated token to the sequence
         input_ids = torch.cat([input_ids, next_token], dim=1)

      return input_ids
   
ds = load_dataset("KomeijiForce/Text2Emoji")
X = ds["train"]['text']

model = Autoencoder(device)

y = model(X[:20])
print(y)
      
      
      