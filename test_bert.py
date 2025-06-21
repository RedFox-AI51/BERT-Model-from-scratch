import torch
import torch.nn.functional as F
from bert_model import (
    create_bert_model, 
    BERTForMaskedLM, 
    BERTForSequenceClassification
)

def test_model_components():
    """Test individual components of the BERT model"""
    print("=" * 50)
    print("Testing BERT Model Components")
    print("=" * 50)
    
    # Create a small model for testing
    config = {
        'hidden_size': 128,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'intermediate_size': 512,
    }
    
    vocab_size = 1000
    model = create_bert_model(vocab_size=vocab_size, config=config)
    
    # Test with different sequence lengths
    test_cases = [
        (1, 10),   # Single sequence, short
        (2, 32),   # Batch of 2, medium length
        (4, 64),   # Batch of 4, longer
    ]
    
    for batch_size, seq_len in test_cases:
        print(f"\nTesting batch_size={batch_size}, seq_len={seq_len}")
        
        # Create random input
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        # Forward pass
        with torch.no_grad():
            sequence_output, pooled_output, _ = model(input_ids, attention_mask)
            
        print(f"  Input: {input_ids.shape}")
        print(f"  Sequence output: {sequence_output.shape}")
        print(f"  Pooled output: {pooled_output.shape}")
        print(f"  Memory usage: {torch.cuda.memory_allocated() / 1024**2:.1f} MB" if torch.cuda.is_available() else "  CPU mode")

def test_attention_patterns():
    """Test attention patterns with a simple example"""
    print("\n" + "=" * 50)
    print("Testing Attention Patterns")
    print("=" * 50)
    
    config = {
        'hidden_size': 64,
        'num_hidden_layers': 1,
        'num_attention_heads': 2,
        'intermediate_size': 256,
    }
    
    model = create_bert_model(vocab_size=100, config=config)
    
    # Create a simple sequence
    input_ids = torch.tensor([[10, 20, 30, 40, 50]])  # Shape: (1, 5)
    attention_mask = torch.ones(1, 5)
    
    # Get attention weights
    with torch.no_grad():
        sequence_output, pooled_output, attentions = model(
            input_ids, attention_mask, output_attentions=True
        )
    
    print(f"Number of attention layers: {len(attentions)}")
    print(f"Attention shape per layer: {attentions[0].shape}")  # (batch, heads, seq, seq)
    
    # Show attention weights for first head of first layer
    attention_weights = attentions[0][0, 0].numpy()  # First batch, first head
    print(f"\nAttention weights (first head):")
    print(attention_weights)

def test_masked_language_modeling():
    """Test the MLM head"""
    print("\n" + "=" * 50)
    print("Testing Masked Language Modeling")
    print("=" * 50)
    
    config = {
        'hidden_size': 128,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'intermediate_size': 512,
    }
    
    vocab_size = 1000
    base_model = create_bert_model(vocab_size=vocab_size, config=config)
    mlm_model = BERTForMaskedLM(base_model, vocab_size)
    
    # Create input with some masked tokens
    input_ids = torch.tensor([[101, 2054, 103, 2003, 103, 102]])  # [CLS] what [MASK] is [MASK] [SEP]
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, 2023, -100, 2146, -100]])  # Only predict masked tokens
    
    # Forward pass
    outputs = mlm_model(input_ids, attention_mask, labels=labels)
    
    print(f"MLM logits shape: {outputs['logits'].shape}")
    if 'loss' in outputs:
        print(f"MLM loss: {outputs['loss'].item():.4f}")
    
    # Get predictions for masked tokens
    predictions = torch.argmax(outputs['logits'], dim=-1)
    print(f"Input tokens: {input_ids.tolist()}")
    print(f"Predicted tokens: {predictions.tolist()}")

def test_sequence_classification():
    """Test sequence classification"""
    print("\n" + "=" * 50)
    print("Testing Sequence Classification")
    print("=" * 50)
    
    config = {
        'hidden_size': 128,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'intermediate_size': 512,
    }
    
    vocab_size = 1000
    num_labels = 3  # 3-class classification
    
    base_model = create_bert_model(vocab_size=vocab_size, config=config)
    cls_model = BERTForSequenceClassification(base_model, num_labels)
    
    # Create batch of sequences
    input_ids = torch.randint(0, vocab_size, (4, 32))  # 4 sequences of length 32
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, num_labels, (4,))  # Random labels
    
    # Forward pass
    outputs = cls_model(input_ids, attention_mask, labels=labels)
    
    print(f"Classification logits shape: {outputs['logits'].shape}")
    print(f"Classification loss: {outputs['loss'].item():.4f}")
    
    # Get predictions
    predictions = torch.argmax(outputs['logits'], dim=-1)
    probabilities = F.softmax(outputs['logits'], dim=-1)
    
    print(f"True labels: {labels.tolist()}")
    print(f"Predictions: {predictions.tolist()}")
    print(f"Prediction probabilities:")
    for i, prob in enumerate(probabilities):
        print(f"  Sample {i}: {prob.tolist()}")

def test_gradient_flow():
    """Test that gradients flow properly"""
    print("\n" + "=" * 50)
    print("Testing Gradient Flow")
    print("=" * 50)
    
    config = {
        'hidden_size': 64,
        'num_hidden_layers': 1,
        'num_attention_heads': 2,
        'intermediate_size': 256,
    }
    
    model = create_bert_model(vocab_size=100, config=config)
    
    # Create dummy input and target
    input_ids = torch.randint(0, 100, (2, 10))
    attention_mask = torch.ones_like(input_ids)
    
    # Forward pass
    sequence_output, pooled_output, _ = model(input_ids, attention_mask)
    
    # Create a dummy loss (sum of all outputs)
    loss = sequence_output.sum() + pooled_output.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_count = 0
    total_grad_norm = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_count += 1
            total_grad_norm += param.grad.norm().item()
    
    print(f"Parameters with gradients: {grad_count}")
    print(f"Total gradient norm: {total_grad_norm:.4f}")
    print("Gradient flow test passed!" if grad_count > 0 else "Warning: No gradients found!")

def benchmark_model():
    """Benchmark model performance"""
    print("\n" + "=" * 50)
    print("Benchmarking Model Performance")
    print("=" * 50)
    
    import time
    
    config = {
        'hidden_size': 256,
        'num_hidden_layers': 4,
        'num_attention_heads': 8,
        'intermediate_size': 1024,
    }
    
    model = create_bert_model(vocab_size=30000, config=config)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    
    # Benchmark different batch sizes
    batch_sizes = [1, 4, 8, 16]
    seq_len = 128
    
    for batch_size in batch_sizes:
        input_ids = torch.randint(0, 30000, (batch_size, seq_len)).to(device)
        attention_mask = torch.ones_like(input_ids)
        
        # Warmup
        with torch.no_grad():
            for _ in range(3):
                model(input_ids, attention_mask)
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(10):
                model(input_ids, attention_mask)
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        print(f"Batch size {batch_size:2d}: {avg_time*1000:.2f} ms per forward pass")

if __name__ == "__main__":
    # Run all tests
    test_model_components()
    test_attention_patterns()
    test_masked_language_modeling()
    test_sequence_classification()
    test_gradient_flow()
    benchmark_model()
    
    print("\n" + "=" * 50)
    print("All tests completed successfully!")
    print("=" * 50)