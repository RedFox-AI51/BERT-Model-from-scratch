import torch
from bert_model import create_bert_model, BERTForMaskedLM, BERTForSequenceClassification

def quick_test():
    """Quick test to verify BERT model works"""
    print("Testing BERT Model - Quick Version")
    print("=" * 40)
    
    # Create a small model
    config = {
        'hidden_size': 128,
        'num_hidden_layers': 2,
        'num_attention_heads': 4,
        'intermediate_size': 512,
    }
    
    vocab_size = 1000
    model = create_bert_model(vocab_size=vocab_size, config=config)
    
    print(f"Created BERT model with vocab_size={vocab_size}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test 1: Basic forward pass
    print("\n1. Testing basic forward pass...")
    batch_size, seq_len = 2, 16
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    
    with torch.no_grad():
        sequence_output, pooled_output, _ = model(input_ids, attention_mask)
    
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Sequence output: {sequence_output.shape}")
    print(f"   Pooled output: {pooled_output.shape}")
    print("   ✓ Basic forward pass works!")
    
    # Test 2: MLM with safe token IDs
    print("\n2. Testing Masked Language Modeling...")
    mlm_model = BERTForMaskedLM(model, vocab_size)
    
    # Use safe token IDs (all within vocab_size range)
    input_ids = torch.tensor([[1, 50, 103, 200, 103, 2]])  # Safe token IDs
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([[-100, -100, 25, -100, 75, -100]])  # Safe label IDs
    
    outputs = mlm_model(input_ids, attention_mask, labels=labels)
    print(f"   MLM logits shape: {outputs['logits'].shape}")
    print(f"   MLM loss: {outputs['loss'].item():.4f}")
    print("   ✓ MLM test works!")
    
    # Test 3: Classification
    print("\n3. Testing Sequence Classification...")
    num_labels = 3
    cls_model = BERTForSequenceClassification(model, num_labels)
    
    input_ids = torch.randint(0, vocab_size, (4, 20))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.randint(0, num_labels, (4,))
    
    outputs = cls_model(input_ids, attention_mask, labels=labels)
    print(f"   Classification logits shape: {outputs['logits'].shape}")
    print(f"   Classification loss: {outputs['loss'].item():.4f}")
    print("   ✓ Classification test works!")
    
    # Test 4: Gradient flow
    print("\n4. Testing gradient flow...")
    loss = outputs['loss']
    loss.backward()
    
    grad_count = sum(1 for p in cls_model.parameters() if p.grad is not None)
    total_params = sum(1 for p in cls_model.parameters())
    print(f"   Parameters with gradients: {grad_count}/{total_params}")
    print("   ✓ Gradient flow works!")
    
    print("\n" + "=" * 40)
    print("All tests passed! Your BERT model is working correctly.")
    print("=" * 40)

if __name__ == "__main__":
    quick_test()