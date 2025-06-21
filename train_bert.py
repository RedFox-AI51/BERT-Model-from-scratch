import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import os
import json
import time
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

from bert_model import (
    create_bert_model, 
    BERTForMaskedLM, 
    BERTForSequenceClassification
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model config
    vocab_size: int = 30000
    hidden_size: int = 256
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 1024
    max_position_embeddings: int = 512
    
    # Training config
    batch_size: int = 8
    learning_rate: float = 5e-5
    num_epochs: int = 3
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Data config
    max_seq_length: int = 128
    mlm_probability: float = 0.15
    
    # Paths
    output_dir: str = "./bert_output"
    save_steps: int = 1000
    logging_steps: int = 100
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class SimpleTextDataset(Dataset):
    """Simple dataset for demonstration - replace with your actual data"""
    
    def __init__(self, texts: List[str], tokenizer=None, max_length: int = 128):
        self.texts = texts
        self.max_length = max_length
        self.tokenizer = tokenizer
        
        # Simple word-level tokenization if no tokenizer provided
        if tokenizer is None:
            self.vocab = self._build_vocab(texts)
            self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
            self.id_to_word = {i: word for word, i in self.word_to_id.items()}
        
    def _build_vocab(self, texts: List[str]) -> List[str]:
        """Build simple vocabulary from texts"""
        vocab = set()
        for text in texts:
            words = text.lower().split()
            vocab.update(words)
        
        # Add special tokens
        special_tokens = ['[PAD]', '[CLS]', '[SEP]', '[MASK]', '[UNK]']
        vocab_list = special_tokens + sorted(list(vocab))
        return vocab_list[:30000]  # Limit vocab size
    
    def _tokenize(self, text: str) -> List[int]:
        """Simple tokenization"""
        words = text.lower().split()
        tokens = [self.word_to_id.get('[CLS]', 1)]  # Start with [CLS]
        
        for word in words:
            token_id = self.word_to_id.get(word, self.word_to_id.get('[UNK]', 4))
            tokens.append(token_id)
            
            if len(tokens) >= self.max_length - 1:  # Leave room for [SEP]
                break
                
        tokens.append(self.word_to_id.get('[SEP]', 2))  # End with [SEP]
        
        # Pad to max_length
        while len(tokens) < self.max_length:
            tokens.append(self.word_to_id.get('[PAD]', 0))
            
        return tokens[:self.max_length]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        input_ids = torch.tensor(self._tokenize(text), dtype=torch.long)
        attention_mask = (input_ids != 0).long()  # Not padding
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'text': text
        }

class MLMDataset(Dataset):
    """Dataset for Masked Language Modeling"""
    
    def __init__(self, base_dataset: SimpleTextDataset, mlm_probability: float = 0.15):
        self.base_dataset = base_dataset
        self.mlm_probability = mlm_probability
        self.mask_token_id = base_dataset.word_to_id.get('[MASK]', 3)
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        input_ids = item['input_ids'].clone()
        labels = input_ids.clone()
        
        # Create random array of floats with same shape as input_ids
        rand = torch.rand(input_ids.shape)
        
        # Create mask array
        mask_arr = (rand < self.mlm_probability) & (input_ids != 0) & (input_ids != 1) & (input_ids != 2)
        
        # Set labels to -100 for non-masked tokens (ignore in loss)
        labels[~mask_arr] = -100
        
        # 80% of the time, replace with [MASK]
        mask_indices = mask_arr & (torch.rand(input_ids.shape) < 0.8)
        input_ids[mask_indices] = self.mask_token_id
        
        # 10% of the time, replace with random token
        random_indices = mask_arr & (torch.rand(input_ids.shape) < 0.5) & (~mask_indices)
        random_tokens = torch.randint(5, len(self.base_dataset.vocab), input_ids.shape, dtype=torch.long)
        input_ids[random_indices] = random_tokens[random_indices]
        
        # 10% of the time, keep original token (no change needed)
        
        return {
            'input_ids': input_ids,
            'attention_mask': item['attention_mask'],
            'labels': labels
        }

class ClassificationDataset(Dataset):
    """Dataset for sequence classification"""
    
    def __init__(self, texts: List[str], labels: List[int], base_dataset: SimpleTextDataset):
        self.texts = texts
        self.labels = labels
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Create a temporary dataset item for tokenization
        temp_dataset = SimpleTextDataset([self.texts[idx]], max_length=self.base_dataset.max_length)
        temp_dataset.vocab = self.base_dataset.vocab
        temp_dataset.word_to_id = self.base_dataset.word_to_id
        temp_dataset.id_to_word = self.base_dataset.id_to_word
        
        item = temp_dataset[0]
        
        return {
            'input_ids': item['input_ids'],
            'attention_mask': item['attention_mask'],
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class BERTTrainer:
    """BERT training class"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize model
        self.base_model = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Training state
        self.global_step = 0
        self.current_loss = 0.0
        
    def create_model(self, task_type: str = "mlm", num_labels: int = 2):
        """Create BERT model for specific task"""
        model_config = {
            'hidden_size': self.config.hidden_size,
            'num_hidden_layers': self.config.num_hidden_layers,
            'num_attention_heads': self.config.num_attention_heads,
            'intermediate_size': self.config.intermediate_size,
            'max_position_embeddings': self.config.max_position_embeddings,
        }
        
        self.base_model = create_bert_model(
            vocab_size=self.config.vocab_size, 
            config=model_config
        )
        
        if task_type == "mlm":
            self.model = BERTForMaskedLM(self.base_model, self.config.vocab_size)
        elif task_type == "classification":
            self.model = BERTForSequenceClassification(self.base_model, num_labels)
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model created: {total_params:,} total parameters, {trainable_params:,} trainable")
        
    def create_optimizer(self, total_steps: int):
        """Create optimizer and scheduler"""
        # Separate weight decay for different parameter types
        param_groups = [
            {
                'params': [p for n, p in self.model.named_parameters() if 'bias' not in n and 'LayerNorm' not in n],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() if 'bias' in n or 'LayerNorm' in n],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = optim.AdamW(param_groups, lr=self.config.learning_rate)
        
        # Linear warmup scheduler
        def lr_lambda(step):
            if step < self.config.warmup_steps:
                return step / max(1, self.config.warmup_steps)
            else:
                return max(0.0, (total_steps - step) / max(1, total_steps - self.config.warmup_steps))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
    def train_mlm(self, dataset: MLMDataset):
        """Train BERT with Masked Language Modeling"""
        logger.info("Starting MLM pre-training...")
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 to avoid multiprocessing issues
        )
        
        total_steps = len(dataloader) * self.config.num_epochs
        self.create_optimizer(total_steps)
        
        self.model.train()
        
        for epoch in range(self.config.num_epochs):
            epoch_loss = 0.0
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs['loss']
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()
                
                # Update tracking
                self.global_step += 1
                epoch_loss += loss.item()
                self.current_loss = loss.item()
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(f"Step {self.global_step}, Loss: {loss.item():.4f}, LR: {lr:.2e}")
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f"checkpoint-{self.global_step}")
            
            avg_loss = epoch_loss / len(dataloader)
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs} - Avg Loss: {avg_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Save final model
        self.save_model("final_mlm_model")
        logger.info("MLM training completed!")
    
    def train_classification(self, train_dataset: ClassificationDataset, eval_dataset: Optional[ClassificationDataset] = None):
        """Fine-tune BERT for classification"""
        logger.info("Starting classification fine-tuning...")
        
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            num_workers=0
        )
        
        total_steps = len(train_dataloader) * self.config.num_epochs
        self.create_optimizer(total_steps)
        
        best_eval_acc = 0.0
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            correct_predictions = 0
            total_predictions = 0
            
            for batch in train_dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs['loss']
                logits = outputs['logits']
                
                # Calculate accuracy
                predictions = torch.argmax(logits, dim=-1)
                correct_predictions += (predictions == batch['labels']).sum().item()
                total_predictions += batch['labels'].size(0)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()
                self.scheduler.step()
                
                epoch_loss += loss.item()
                self.global_step += 1
                
                if self.global_step % self.config.logging_steps == 0:
                    logger.info(f"Step {self.global_step}, Loss: {loss.item():.4f}")
            
            train_acc = correct_predictions / total_predictions
            avg_loss = epoch_loss / len(train_dataloader)
            
            logger.info(f"Epoch {epoch + 1} - Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Evaluation
            if eval_dataset:
                eval_acc = self.evaluate_classification(eval_dataset)
                logger.info(f"Epoch {epoch + 1} - Eval Acc: {eval_acc:.4f}")
                
                if eval_acc > best_eval_acc:
                    best_eval_acc = eval_acc
                    self.save_model("best_classification_model")
        
        self.save_model("final_classification_model")
        logger.info(f"Classification training completed! Best eval accuracy: {best_eval_acc:.4f}")
    
    def evaluate_classification(self, dataset: ClassificationDataset) -> float:
        """Evaluate classification model"""
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0)
        
        self.model.eval()
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=-1)
                
                correct_predictions += (predictions == batch['labels']).sum().item()
                total_predictions += batch['labels'].size(0)
        
        return correct_predictions / total_predictions
    
    def save_model(self, name: str):
        """Save model and config"""
        save_path = os.path.join(self.config.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(save_path, "model.pt"))
        
        # Save config
        config_dict = {
            'vocab_size': self.config.vocab_size,
            'hidden_size': self.config.hidden_size,
            'num_hidden_layers': self.config.num_hidden_layers,
            'num_attention_heads': self.config.num_attention_heads,
            'intermediate_size': self.config.intermediate_size,
            'max_position_embeddings': self.config.max_position_embeddings,
        }
        
        with open(os.path.join(save_path, "config.json"), 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    def save_checkpoint(self, name: str):
        """Save training checkpoint"""
        save_path = os.path.join(self.config.output_dir, name)
        os.makedirs(save_path, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.config.__dict__
        }
        
        torch.save(checkpoint, os.path.join(save_path, "checkpoint.pt"))

def create_sample_data():
    """Create sample data for testing"""
    # Sample texts for pre-training
    pretrain_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Natural language processing is a fascinating field of study.",
        "Machine learning models require large amounts of data to train effectively.",
        "BERT revolutionized the field of natural language understanding.",
        "Transformers use attention mechanisms to process sequential data.",
        "Deep learning has achieved remarkable success in many domains.",
        "Text classification is an important task in NLP applications.",
        "Language models can generate coherent and contextual text.",
        "Pre-training on large corpora improves downstream task performance.",
        "Fine-tuning allows models to adapt to specific tasks and domains."
    ] * 100  # Repeat for more training data
    
    # Sample data for classification
    classification_texts = [
        "This movie is absolutely amazing and I loved every minute of it!",
        "The film was boring and I fell asleep halfway through.",
        "Great acting and wonderful cinematography made this a masterpiece.",
        "Terrible plot and poor character development ruined the experience.",
        "I enjoyed the movie but it could have been better.",
        "Outstanding performance by all actors in this thrilling adventure.",
        "The story was confusing and hard to follow throughout.",
        "Beautiful visuals and excellent soundtrack enhanced the viewing experience.",
        "Disappointing sequel that failed to live up to expectations.",
        "Brilliant writing and direction made this film unforgettable."
    ] * 20  # Repeat for more data
    
    # Labels: 0 = negative, 1 = positive
    classification_labels = [1, 0, 1, 0, 1, 1, 0, 1, 0, 1] * 20
    
    return pretrain_texts, classification_texts, classification_labels

def main():
    """Main training function"""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Configuration
    config = TrainingConfig(
        vocab_size=5000,  # Smaller vocab for demo
        hidden_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        intermediate_size=1024,
        batch_size=4,
        num_epochs=2,
        learning_rate=2e-5,
        max_seq_length=64,
        output_dir="./bert_training_output"
    )
    
    logger.info(f"Using device: {config.device}")
    logger.info(f"Training configuration: {config}")
    
    # Create sample data
    pretrain_texts, classification_texts, classification_labels = create_sample_data()
    
    # Create datasets
    base_dataset = SimpleTextDataset(pretrain_texts, max_length=config.max_seq_length)
    mlm_dataset = MLMDataset(base_dataset, config.mlm_probability)
    
    # Split classification data
    split_idx = int(0.8 * len(classification_texts))
    train_cls_dataset = ClassificationDataset(
        classification_texts[:split_idx], 
        classification_labels[:split_idx], 
        base_dataset
    )
    eval_cls_dataset = ClassificationDataset(
        classification_texts[split_idx:], 
        classification_labels[split_idx:], 
        base_dataset
    )
    
    logger.info(f"Created datasets: {len(mlm_dataset)} MLM samples, {len(train_cls_dataset)} train classification, {len(eval_cls_dataset)} eval classification")
    
    # Phase 1: Pre-training with MLM
    logger.info("="*50)
    logger.info("PHASE 1: PRE-TRAINING WITH MASKED LANGUAGE MODELING")
    logger.info("="*50)
    
    trainer = BERTTrainer(config)
    trainer.create_model(task_type="mlm")
    trainer.train_mlm(mlm_dataset)
    
    # Phase 2: Fine-tuning for classification
    logger.info("="*50)
    logger.info("PHASE 2: FINE-TUNING FOR CLASSIFICATION")
    logger.info("="*50)
    
    # Create new trainer for classification (or reset the existing one)
    config.num_epochs = 3  # More epochs for fine-tuning
    config.learning_rate = 2e-5  # Lower learning rate for fine-tuning
    
    cls_trainer = BERTTrainer(config)
    cls_trainer.create_model(task_type="classification", num_labels=2)
    
    # Load pre-trained weights if available
    try:
        pretrained_path = os.path.join(config.output_dir, "final_mlm_model", "model.pt")
        if os.path.exists(pretrained_path):
            # Load only the BERT base model weights
            pretrained_state = torch.load(pretrained_path, map_location=cls_trainer.device)
            
            # Filter out the MLM head weights and load only BERT weights
            bert_state = {}
            for key, value in pretrained_state.items():
                if key.startswith('bert.'):
                    bert_state[key] = value
            
            cls_trainer.model.load_state_dict(bert_state, strict=False)
            logger.info("Loaded pre-trained BERT weights for fine-tuning")
    except Exception as e:
        logger.warning(f"Could not load pre-trained weights: {e}")
    
    cls_trainer.train_classification(train_cls_dataset, eval_cls_dataset)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()