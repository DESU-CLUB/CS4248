import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class Adapter(nn.Module):
    def __init__(self, encoder_dim, decoder_dim):
        super().__init__()
        self.linear = nn.Linear(encoder_dim, decoder_dim)
        self.layer_norm = nn.LayerNorm(decoder_dim)
        
    def forward(self, x):
        # Handle 2D input (batch_size, hidden_dim)
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        return self.layer_norm(self.linear(x))

class CustomDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_decoder_layers=6,
                 dim_feedforward=2048, dropout=0.1, encoder_dim=128):
        super().__init__()
        
        # Adapter to convert encoder hidden dimensions to decoder dimensions
        self.adapter = Adapter(encoder_dim, d_model)
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer decoder
        decoder_layer = TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def _generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, encoder_outputs, target_tokens=None, tgt_mask=None, tgt_key_padding_mask=None):
        # Process encoder outputs through adapter
        memory = self.adapter(encoder_outputs)
        
        if target_tokens is None:
            # For inference (auto-regressive generation)
            return self.generate(memory)
        
        # Prepare target sequences (shift right for teacher forcing)
        tgt = target_tokens[:, :-1]  # remove last token
        tgt_emb = self.token_embedding(tgt)
        tgt_emb = self.positional_encoding(tgt_emb)
        
        # Create causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(tgt.size(1)).to(tgt_emb.device)
            
        # Decode
        output = self.transformer_decoder(
            tgt_emb, 
            memory, 
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    def generate(self, memory, max_length=50, temperature=1.0, top_k=50):
        """Generate text autogressively using encoder memory."""
        batch_size = memory.size(0)
        device = memory.device
        
        # Start with empty sequence
        input_ids = torch.zeros((batch_size, 1), dtype=torch.long, device=device)
        
        # Generation loop
        generated_tokens = []
        
        for i in range(max_length):
            # Embed input tokens
            tgt_emb = self.token_embedding(input_ids)
            tgt_emb = self.positional_encoding(tgt_emb)
            
            # Create causal mask
            tgt_mask = self._generate_square_subsequent_mask(input_ids.size(1)).to(device)
            
            # Decode
            output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask)
            
            # Get logits for next token (last position)
            next_token_logits = self.output_projection(output[:, -1, :]) / temperature
            
            # Apply top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
                
            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Store token
            generated_tokens.append(next_token)
            
            # Update input sequence for next iteration
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Early stopping if all sequences have end token (could implement this if needed)
            
        return torch.cat(generated_tokens, dim=1)
    
