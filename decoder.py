import torch
import torch.nn as nn
from torch.nn import TransformerDecoder, TransformerDecoderLayer
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

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
    
class LlamaDecoder(nn.Module):
    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct", encoder_dim=2048, device=None):
        """
        LlamaDecoder using Llama 3.2 1B with an adapter for encoder embeddings.
        
        Args:
            model_name: The model name/path for the Llama model
            encoder_dim: Dimension of the encoder embeddings (default 2048)
            device: Device to load the model on
        """
        super().__init__()
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Load model config first to get embedding dimension
        self.config = AutoConfig.from_pretrained(model_name)
        self.model_dim = self.config.hidden_size
        
        # Create embedding adapter (from encoder dimension to Llama hidden size)
        self.adapter = nn.Sequential(
            nn.Linear(encoder_dim, 4 * self.model_dim),
            nn.LayerNorm(4 * self.model_dim),
            nn.GELU(),
            nn.Linear(4 * self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim)
        )
        
        # Load Llama model
        print(f"Loading Llama model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
            device_map=self.device
        )
        
        # Move model components to device
        self.adapter.to(self.device)
        
        # Get vocab size for final projection
        self.vocab_size = self.model.config.vocab_size
        print(f"Llama model loaded with vocab size: {self.vocab_size}")
        
        # Reference to the tokenizer's embedding layer for direct embedding lookup
        self.token_embedding = self.model.get_input_embeddings()
        
    def forward(self, encoder_outputs, input_ids=None, inputs_embeds=None):
        """
        Forward pass through the LlamaDecoder.
        
        This method supports three modes:
        1. With encoder_outputs only: Adapts the encoder outputs and uses them directly
        2. With encoder_outputs + input_ids: Combines adapted encoder embeddings with input token embeddings
        3. With encoder_outputs + inputs_embeds: Combines adapted encoder embeddings with provided embeddings
        
        Args:
            encoder_outputs: Outputs from the encoder (batch_size, encoder_dim)
            input_ids: Optional input token IDs for teacher forcing (batch_size, seq_len)
            inputs_embeds: Optional pre-computed embeddings from another model (batch_size, seq_len, hidden_size)
            
        Returns:
            Logits for next token prediction (batch_size, seq_len, vocab_size)
        """
        batch_size = encoder_outputs.size(0)
        
        # Adapt encoder outputs to match Llama hidden dimension
        adapted_embeddings = self.adapter(encoder_outputs)
        
        # Reshape to (batch_size, 1, hidden_size) to serve as a prefix/conditioning
        # This will be the first token embedding in the sequence
        adapted_embeddings = adapted_embeddings.unsqueeze(1)
        
        # CASE 1: Using pre-computed embeddings from another model
        if inputs_embeds is not None:
            # Ensure inputs_embeds has the correct hidden dimension
            if inputs_embeds.size(-1) != self.model_dim:
                raise ValueError(f"Input embeddings dimension {inputs_embeds.size(-1)} doesn't match model dimension {self.model_dim}")
            
            # Concatenate adapted embeddings with provided embeddings
            combined_embeds = torch.cat([adapted_embeddings, inputs_embeds], dim=1)
            
            # Create attention mask that includes all tokens
            attention_mask = torch.ones(
                (batch_size, combined_embeds.size(1)),
                device=combined_embeds.device
            )
            
            # Forward pass with embeddings directly
            outputs = self.model(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get logits (batch_size, seq_len, vocab_size)
            logits = outputs.logits
            
            # Return logits without the first position (prefix token)
            return logits[:, 1:, :]
        
        # CASE 2: Using input_ids for embedding lookup
        elif input_ids is not None:
            # Get embeddings for input_ids via the model's embedding layer
            input_embeds = self.token_embedding(input_ids)
            
            # Concatenate adapted embeddings with input embeddings
            combined_embeds = torch.cat([adapted_embeddings, input_embeds], dim=1)
            
            # Create attention mask that includes the prefix token
            attention_mask = torch.ones(
                (batch_size, combined_embeds.size(1)),
                device=combined_embeds.device
            )
            
            # Forward pass with embeddings directly
            outputs = self.model(
                inputs_embeds=combined_embeds,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Get logits (batch_size, seq_len, vocab_size)
            logits = outputs.logits
            
            # Return logits without the first position (prefix token)
            return logits[:, 1:, :]
        
        # CASE 3: Using only encoder outputs
        else:
            # Create attention mask for a single token
            attention_mask = torch.ones((batch_size, 1), device=self.device)
            
            # Forward pass with just the adapted embeddings
            outputs = self.model(
                inputs_embeds=adapted_embeddings,
                attention_mask=attention_mask,
                return_dict=True
            )
            
            # Return logits
            return outputs.logits
    
    def generate(self, encoder_outputs, max_length=50, temperature=0.7, top_p=0.9, inputs_embeds=None):
        """
        Generate text from encoder outputs, optionally using initial embeddings.
        
        Args:
            encoder_outputs: Encoder embeddings (batch_size, encoder_dim)
            max_length: Maximum length of generated sequence
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            inputs_embeds: Optional initial embeddings from another model
            
        Returns:
            Generated token IDs
        """
        batch_size = encoder_outputs.size(0)
        
        # Adapt encoder outputs
        adapted_embeddings = self.adapter(encoder_outputs)
        adapted_embeddings = adapted_embeddings.unsqueeze(1)
        
        # Initialize embeddings sequence
        if inputs_embeds is not None:
            # Start with adapted embeddings + provided embeddings
            current_embeds = torch.cat([adapted_embeddings, inputs_embeds], dim=1)
        else:
            # Start with just adapted embeddings
            current_embeds = adapted_embeddings
        
        # Start with an empty tensor for generated ids
        generated_ids = torch.zeros((batch_size, 0), dtype=torch.long, device=self.device)
        
        # Generation loop
        for _ in range(max_length):
            # Create appropriate attention mask
            attention_mask = torch.ones((batch_size, current_embeds.size(1)), device=self.device)
            
            # Forward pass with embeddings directly
            with torch.no_grad():
                outputs = self.model(
                    inputs_embeds=current_embeds,
                    attention_mask=attention_mask,
                    return_dict=True
                )
            
            # Get next token logits (last position)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temperature
            
            # Apply top-p (nucleus) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # Scatter sorted indices to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            next_token_logits[indices_to_remove] = -float('Inf')
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated ids
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Get embeddings for next token
            next_token_embeds = self.token_embedding(next_token)
            
            # Append to current embeddings
            current_embeds = torch.cat([current_embeds, next_token_embeds], dim=1)
            
        return generated_ids
    


