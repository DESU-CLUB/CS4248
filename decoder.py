from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn

model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

class CrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, y):
        # MaskedCrossAttention
        batch_size = x.size(0)
        q = self.q_proj(x)
        k = self.k_proj(y)
        v = self.v_proj(y)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Create attention mask (you can customize this based on your requirements)
        # For example, creating a causal mask where each position can only attend to previous positions
        seq_len = k.size(2)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        mask = mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions
        
        # Apply mask by setting masked positions to -inf before softmax
        attn_weights = attn_weights.masked_fill(mask, float('-inf'))
        
        # Apply softmax to get attention probabilities
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape back to original dimensions
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # Final projection
        out = self.out_proj(out)
        return out
        
#Cross Attention for LLaMA to translate embeddings from concept model to text
class LlamaWithCrossAttention(nn.Module):
    def __init__(self, llama_model, hidden_dim, num_heads):
        super().__init__()
        self.llama = llama_model
        
        # Get the number of layers in the LLaMA model
        num_layers = len(self.llama.model.layers)
        
        # Create cross-attention modules for each layer
        self.cross_attentions = nn.ModuleList([
            CrossAttention(hidden_dim, num_heads) for _ in range(num_layers)
        ])
        
        # Layer norms and projections for each layer
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.fcs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
    def forward(self, input_ids, attention_mask=None, context=None):
        # Start with embedding lookup
        hidden_states = self.llama.model.embed_tokens(input_ids)
        
        # Process through each layer with cross-attention
        for i, layer in enumerate(self.llama.model.layers):
            # Apply LLaMA layer's self-attention and feed-forward
            layer_outputs = layer(hidden_states, attention_mask=attention_mask)
            hidden_states = layer_outputs[0]
            
            # Apply cross-attention if context is provided
            if context is not None:
                # Apply cross-attention between LLaMA hidden states and context
                cross_attn_output = self.cross_attentions[i](hidden_states, context)
                
                # Residual connection and normalization
                hidden_states = hidden_states + cross_attn_output
                hidden_states = self.norms[i](hidden_states)
                hidden_states = self.fcs[i](hidden_states)
        
        # Final normalization
        hidden_states = self.llama.model.norm(hidden_states)
        
        # For language modeling
        logits = self.llama.lm_head(hidden_states)
        
        return logits


        
        
        