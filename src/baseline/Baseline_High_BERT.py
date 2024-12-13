import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
import pandas as pd

MAX_LENGTH = 512

data = pd.read_csv("data/IMDB.csv")

text_long = data['review'].tolist()
labels_long = data['sentiment'].tolist()

print("data loaded")

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


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
        encoding = self.tokenizer(text, return_tensors='pt', max_length=MAX_LENGTH, padding='max_length', truncation=True)

        return {'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'label': torch.tensor(label, dtype=torch.long)}

class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

def train(model, data_loader, optimizer, scheduler, device):
    i=1
    model.train()
    for batch in data_loader:
        if(i%10==0):
            print("batch ", i)
        i+=1
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
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
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return classification_report(actual_labels, predictions, output_dict=True)

bert_model_name = 'bert-base-uncased'
num_classes = 2
batch_size = 32
num_epochs = 4
learning_rate = 2e-5

print("start loading")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("start training")

num_fold = 5
kf = KFold(n_splits=num_fold, shuffle=True, random_state=42)
Report = [[None] * num_epochs for _ in range(num_fold)]

for fold, (train_idx, val_idx) in enumerate(kf.split(text_long)):
    print(f"Fold {fold + 1}")
    
    model = BERTClassifier(bert_model_name, num_classes).to(device)
    
    train_texts = [text_long[i] for i in train_idx]
    val_texts = [text_long[i] for i in val_idx]
    train_labels = [labels_long[i] for i in train_idx]
    val_labels = [labels_long[i] for i in val_idx]
    
    train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        train(model, train_dataloader, optimizer, scheduler, device)
        report = evaluate(model, val_dataloader, device)
        print(report)
        print("\n")
        Report[fold][epoch] = report

filename = "Result_Baseline_High_BERT.txt"

with open(filename, "w", encoding="utf-8") as f:
    for fold in range(len(Report)):
        f.write(f"Fold {fold+1}" + "\n")
        for epoch in range(len(Report[fold])):
            f.write(f"Epoch {epoch+1}" + "\n")
            for line in Report[fold][epoch]:
                f.write(f"{line}: {Report[fold][epoch][line]}" + "\n")
            f.write("\n")
        f.write("\n\n")

print(f"Report saved to {filename}")