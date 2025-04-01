from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

class Adapter(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.linear = nn.Linear(model.config.hidden_size, 512)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(512, model.config.hidden_size)

    def forward(self, x):
        return self.linear2(self.gelu(self.linear(x)))
    

class LLamaWithAdapter(nn.Module):
    def __init__(self, model, adapter):
        super().__init__()
        self.model = model
        self.adapter = adapter
    
    def forward(self, embeddings, attention_mask=None):
        # Process embeddings through adapter
        adapter_output = self.adapter(embeddings)
        
        # Add adapter output to original embeddings (residual connection)
        hidden_states = adapter_output
        
        # Process through LLaMA model layers
        for layer in self.model.model.layers:
            outputs = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = outputs[0]
        
        # Final normalization
        hidden_states = self.model.model.norm(hidden_states)
        
        # Get logits
        logits = self.model.lm_head(hidden_states)
        
        return logits
    
    def generate(self, embeddings, max_length=50, temperature=1.0, top_k=50):
        """
        Generate text autogressively using embeddings as initial context.
        
        Args:
            embeddings: The input embeddings (batch_size, seq_len, hidden_size)
            max_length: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Number of highest probability tokens to keep for top-k sampling
            
        Returns:
            Generated token IDs (batch_size, generated_seq_len)
        """
        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # Initialize with a start token
        input_ids = torch.full((batch_size, 1), 
                               self.model.config.bos_token_id, 
                               device=device)
        
        # Convert input_ids to embeddings
        current_embeddings = self.model.model.embed_tokens(input_ids)
        
        # Add adapter output to the start embeddings
        adapter_output = self.adapter(embeddings)
        modified_embeddings = current_embeddings + adapter_output
        
        # Generate tokens one by one
        generated_ids = []
        
        for _ in range(max_length):
            # Forward pass through model
            logits = self.forward(modified_embeddings)
            
            # Get next token logits (last position)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k sampling
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = float('-inf')
                
            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids.append(next_token)
            
            # Get embedding for the new token
            next_embedding = self.model.model.embed_tokens(next_token)
            
            # Concatenate with current embeddings
            modified_embeddings = torch.cat([modified_embeddings, next_embedding], dim=1)
            
            # Check for EOS
            if (next_token == self.model.config.eos_token_id).all():
                break
                
        return torch.cat(generated_ids, dim=1)
    
