import os
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertTokenizer, 
    BeitModel, 
    BeitConfig, 
    BertModel, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import pandas as pd

MAX_LENGTH = 197

# Load pre-trained BERT tokenizer
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# Initialize word embeddings from BERT
bert_model = BertModel.from_pretrained(model_name)
vocab = tokenizer.get_vocab()
embedding_layer = bert_model.embeddings.word_embeddings
embedding_weights = embedding_layer.weight.data  # This contains all word embeddings

# Create a dictionary of word to embedding
word_embeddings = {word: embedding_weights[idx].cpu().numpy() for word, idx in vocab.items()}

# Load dataset
data = pd.read_csv("data/IMDB.csv")

text_long = data['review'].tolist()
labels_long = data['sentiment'].tolist()

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if idx >= len(self.labels) or idx >= len(self.texts):
            raise IndexError(f"Index {idx} out of bounds for dataset with length {len(self.labels)}")
        text = self.texts[idx]
        label = self.labels[idx]
        inputs = self.tokenizer(
            text, 
            add_special_tokens=True, 
            max_length=MAX_LENGTH, 
            padding='max_length', 
            truncation=True,
            return_tensors='pt'
        )
        tokens_with_special = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'].squeeze())
        result = []
        for token in tokens_with_special:
            # Handle unknown tokens
            if token in word_embeddings:
                result.append(word_embeddings[token])
            else:
                result.append(np.zeros(bert_model.config.hidden_size))
        result = np.array(result)

        return {
            'tensor': torch.from_numpy(result).float(), 
            'label': torch.tensor(label).long()
        }

class ModifiedBEiT(nn.Module):
    def __init__(self, original_model, resolution):
        super().__init__()
        self.original_model = original_model
        self.resolution = resolution  # keep resolution

    def forward(self, x):
        # x: [batch_size, seq_len, hidden_dim] 
        encoder_outputs = self.original_model.encoder(
            hidden_states=x,
            resolution=self.resolution,  
            return_dict=True
        )
        return encoder_outputs.last_hidden_state  

# BEiTClassifier
class NLPBEiT(nn.Module):
    def __init__(self, beit_model_name, num_classes, resolution=(224, 224)):
        super(NLPBEiT, self).__init__()
        # Initialize BEiT config
        config = BeitConfig.from_pretrained(beit_model_name)
        
        # Initialize BEiT model with random weights
        beit_model = BeitModel(config)  # ランダム初期化
        
        # ModifiedBEiT
        self.beit = ModifiedBEiT(beit_model, resolution)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(config.hidden_size, num_classes)

    def forward(self, hidden_states):
        last_hidden_state = self.beit(hidden_states)
        cls_token = last_hidden_state[:, 0, :]  # extract [CLS] token
        cls_token = cls_token.contiguous()  
        x = self.dropout(cls_token)
        logits = self.fc(x)
        return logits

def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    for i, batch in enumerate(data_loader, 1):
        if i % 10 == 0:
            print("batch ", i)
        optimizer.zero_grad()
        tensor = batch['tensor'].to(device)
        labels = batch['label'].to(device)
        outputs = model(tensor)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            tensor = batch['tensor'].to(device)
            labels = batch['label'].to(device)
            outputs = model(tensor)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return classification_report(actual_labels, predictions, output_dict=True)

# Hyperparameters
bert_model_name = 'bert-base-uncased'
beit_model_name = "microsoft/beit-base-patch16-224-pt22k-ft22k"
num_classes = 2
batch_size = 32
num_epochs = 4
learning_rate = 2e-5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
num_fold = 5
kf = KFold(n_splits=num_fold, shuffle=True, random_state=42)
Report = [[None] * num_epochs for _ in range(num_fold)]

for fold, (train_idx, val_idx) in enumerate(kf.split(text_long)):
    print(f"Fold {fold + 1}")

    model = NLPBEiT(beit_model_name, num_classes).to(device)

    # Split data
    train_texts = [text_long[i] for i in train_idx]
    val_texts = [text_long[i] for i in val_idx]
    train_labels = [labels_long[i] for i in train_idx]
    val_labels = [labels_long[i] for i in val_idx]

    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0) 

    # Prepare for the model
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )

    # Train the model
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_dataloader, optimizer, scheduler, device)
        report = evaluate(model, val_dataloader, device)
        print(report)
        print("\n")
        Report[fold][epoch] = report
        
# Save the results
filename = "Result_NLP_BEiT_Ini.txt"

with open(filename, "w", encoding="utf-8") as f:
    for fold in range(len(Report)):
        f.write(f"Fold {fold+1}\n")
        for epoch in range(len(Report[fold])):
            f.write(f"Epoch {epoch+1}\n")
            for key, value in Report[fold][epoch].items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        f.write("\n\n")

print(f"Report saved to {filename}")