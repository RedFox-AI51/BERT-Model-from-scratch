# 🧠 BERT Model from Scratch

This project implements a BERT (Bidirectional Encoder Representations from Transformers) model **entirely from scratch** using PyTorch, including tokenization, model architecture, and training pipeline.

## 🚀 Features

- Custom tokenizer with vocabulary building
- BERT model implementation (multi-layer Transformer encoder)
- Masked Language Modeling (MLM) training
- Evaluation metrics & logging
- Configurable architecture via `TrainingConfig`

## 🛠️ Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🧪 Training the Model

To start training:

```bash
python train_bert.py
```

Training settings can be modified in the script:
- `vocab_size`
- `hidden_size`
- `num_hidden_layers`
- `num_attention_heads`
- `intermediate_size`
- `max_position_embeddings`

## 📊 Example Output

Training logs include:
- Loss
- Learning rate
- Training accuracy
- Evaluation accuracy

Graph your logs using tools like Desmos, Matplotlib, or TensorBoard.

## 📁 Project Structure

```
BERT-Model-from-scratch/
├── tokenizer.py        # Custom tokenizer and vocab
├── bert_model.py       # BERT model definition
├── train_bert.py       # Training script
├── quick_test.py       # Perform a quick test script
├── test_bert.py        # Perform a regular test script
├── utils.py            # Data prep, batching, and helpers
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## 📜 License

This project is licensed under the [MIT License](LICENSE).

## 🙋‍♂️ Author

**RedFox-AI51**  
If you like this project, feel free to ⭐ the repo or contribute!
