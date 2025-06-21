import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple

class BERTEmbeddings(nn.Module):
    """
    BERT Embeddings: Token + Position + Segment Embeddings
    """
    def __init__(self, vocab_size: int, hidden_size: int, max_position_embeddings: int = 512, 
                 type_vocab_size: int = 2, dropout_prob: float = 0.1):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        
        # Register position_ids as buffer (not a parameter)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))
        
    def forward(self, input_ids: torch.Tensor, token_type_ids: Optional[torch.Tensor] = None, 
                position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        seq_length = input_ids.size(1)
        
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]
            
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
            
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        token_type_embeds = self.token_type_embeddings(token_type_ids)
        
        embeddings = token_embeds + position_embeds + token_type_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism
    """
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float = 0.1):
        super().__init__()
        assert hidden_size % num_attention_heads == 0
        
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout_prob)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Calculate attention scores
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
            
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        return context_layer, attention_probs

class BERTSelfAttention(nn.Module):
    """
    BERT Self-Attention with output projection
    """
    def __init__(self, hidden_size: int, num_attention_heads: int, dropout_prob: float = 0.1):
        super().__init__()
        self.self = MultiHeadSelfAttention(hidden_size, num_attention_heads, dropout_prob)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        self_outputs, attention_probs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs)
        attention_output = self.dropout(attention_output)
        return attention_output, attention_probs

class BERTIntermediate(nn.Module):
    """
    Feed-forward intermediate layer
    """
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, intermediate_size)
        self.intermediate_act_fn = nn.GELU()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BERTOutput(nn.Module):
    """
    Output layer with residual connection and layer norm
    """
    def __init__(self, intermediate_size: int, hidden_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states

class BERTLayer(nn.Module):
    """
    Single BERT transformer layer
    """
    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.attention = BERTSelfAttention(hidden_size, num_attention_heads, dropout_prob)
        self.intermediate = BERTIntermediate(hidden_size, intermediate_size)
        self.output = BERTOutput(intermediate_size, hidden_size, dropout_prob)
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)
        attention_output = self.attention_layer_norm(attention_output + hidden_states)
        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        
        return layer_output, attention_probs

class BERTEncoder(nn.Module):
    """
    Stack of BERT layers
    """
    def __init__(self, num_hidden_layers: int, hidden_size: int, num_attention_heads: int, 
                 intermediate_size: int, dropout_prob: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            BERTLayer(hidden_size, num_attention_heads, intermediate_size, dropout_prob)
            for _ in range(num_hidden_layers)
        ])
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, 
                output_attentions: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        all_attentions = () if output_attentions else None
        
        for layer in self.layers:
            hidden_states, attention_probs = layer(hidden_states, attention_mask)
            if output_attentions:
                all_attentions = all_attentions + (attention_probs,)
                
        return hidden_states, all_attentions

class BERTPooler(nn.Module):
    """
    Pooler for classification tasks (uses [CLS] token)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Take the hidden state corresponding to the first token ([CLS])
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BERTModel(nn.Module):
    """
    Main BERT model
    """
    def __init__(self, vocab_size: int, hidden_size: int = 768, num_hidden_layers: int = 12,
                 num_attention_heads: int = 12, intermediate_size: int = 3072,
                 max_position_embeddings: int = 512, type_vocab_size: int = 2,
                 dropout_prob: float = 0.1):
        super().__init__()
        
        self.embeddings = BERTEmbeddings(
            vocab_size, hidden_size, max_position_embeddings, type_vocab_size, dropout_prob
        )
        
        self.encoder = BERTEncoder(
            num_hidden_layers, hidden_size, num_attention_heads, intermediate_size, dropout_prob
        )
        
        self.pooler = BERTPooler(hidden_size)
        
    def get_extended_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Convert attention mask to format expected by attention layers
        """
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
            
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        return extended_attention_mask
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None, position_ids: Optional[torch.Tensor] = None,
                output_attentions: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, ...]]]:
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
            
        extended_attention_mask = self.get_extended_attention_mask(attention_mask)
        
        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)
        
        encoder_outputs, all_attentions = self.encoder(
            embedding_output, extended_attention_mask, output_attentions
        )
        
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output)
        
        return sequence_output, pooled_output, all_attentions

class BERTForMaskedLM(nn.Module):
    """
    BERT model for Masked Language Modeling
    """
    def __init__(self, bert_model: BERTModel, vocab_size: int):
        super().__init__()
        self.bert = bert_model
        self.cls = nn.Linear(bert_model.embeddings.token_embeddings.embedding_dim, vocab_size)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> dict:
        
        sequence_output, pooled_output, _ = self.bert(input_ids, attention_mask, token_type_ids)
        
        prediction_scores = self.cls(sequence_output)
        
        outputs = {"logits": prediction_scores}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, prediction_scores.size(-1)), labels.view(-1))
            outputs["loss"] = masked_lm_loss
            
        return outputs

class BERTForSequenceClassification(nn.Module):
    """
    BERT model for sequence classification
    """
    def __init__(self, bert_model: BERTModel, num_labels: int):
        super().__init__()
        self.bert = bert_model
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(bert_model.pooler.dense.out_features, num_labels)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None, labels: Optional[torch.Tensor] = None) -> dict:
        
        sequence_output, pooled_output, _ = self.bert(input_ids, attention_mask, token_type_ids)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = {"logits": logits}
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
            outputs["loss"] = loss
            
        return outputs

# Example usage and testing
def create_bert_model(vocab_size: int = 30522, config: dict = None) -> BERTModel:
    """
    Create a BERT model with default or custom configuration
    """
    default_config = {
        'hidden_size': 768,
        'num_hidden_layers': 12,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'dropout_prob': 0.1
    }
    
    if config:
        default_config.update(config)
        
    return BERTModel(vocab_size=vocab_size, **default_config)

def test_bert_model():
    """
    Test the BERT model with dummy data
    """
    # Create a small BERT model for testing
    config = {
        'hidden_size': 256,
        'num_hidden_layers': 4,
        'num_attention_heads': 8,
        'intermediate_size': 1024,
    }
    
    model = create_bert_model(vocab_size=1000, config=config)
    
    # Create dummy input
    batch_size, seq_len = 2, 64
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
    
    # Forward pass
    with torch.no_grad():
        sequence_output, pooled_output, _ = model(input_ids, attention_mask, token_type_ids)
        
    print(f"Input shape: {input_ids.shape}")
    print(f"Sequence output shape: {sequence_output.shape}")
    print(f"Pooled output shape: {pooled_output.shape}")
    print("BERT model test passed!")
    
    # Test MLM head
    mlm_model = BERTForMaskedLM(model, vocab_size=1000)
    mlm_outputs = mlm_model(input_ids, attention_mask, token_type_ids)
    print(f"MLM logits shape: {mlm_outputs['logits'].shape}")
    
    # Test classification head
    cls_model = BERTForSequenceClassification(model, num_labels=2)
    cls_outputs = cls_model(input_ids, attention_mask, token_type_ids)
    print(f"Classification logits shape: {cls_outputs['logits'].shape}")

if __name__ == "__main__":
    test_bert_model()