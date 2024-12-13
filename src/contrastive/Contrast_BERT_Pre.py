import os
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BeitImageProcessor,  get_linear_schedule_with_warmup
from transformers import BertModel
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from PIL import Image

df = pd.read_csv("image_embeddings_768d.csv")
df['file_path'] = df['file_path'].str.replace('origin/Dog_Cat/', '', regex=False)
df['label'] = df['file_path'].apply(
    lambda x: 0 if 'cat' in x.lower() else (1 if 'dog' in x.lower() else None)
)

image_labels = torch.tensor(df['label'])
image_features = torch.tensor(df.drop(columns = ['file_path','label'], axis=1).values, dtype=float)
image_features = image_features.float()


class BeitDataset(Dataset):
    def __init__(self, image_vec, labels):
        self.image_vec = image_vec
        self.labels = labels

    def __len__(self):
        return len(self.image_vec)

    def __getitem__(self, idx):
        img_vec = self.image_vec[idx]
        label = torch.tensor(self.labels[idx])
        return {
            "img_vec": img_vec,
            "label": label
        }
        

class CVBERT(nn.Module):
    def __init__(self, bert_model_name, embed_dim=768, num_classes=2):
        super(CVBERT, self).__init__()
        
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        self.bert.embeddings = CustomEmbeddingLayer(self.bert.embeddings)
        
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_tensor, attention_mask=None):
        """
        input_tensor: [batch_size, 768]
        """
        batch_size, embed_dim = input_tensor.size()
        
        # [batch_size, sequence_length, hidden_size], sequence_length=1
        inputs_embeds = input_tensor.unsqueeze(1)  # [batch_size, 1, 768]
        
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, 1, device=input_tensor.device)
        
        # BERTに入力
        outputs = self.bert(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            input_ids=None
        )
        pooled_output = outputs.pooler_output  # [batch_size, hidden_size]
        
        x = self.dropout(pooled_output)
        output = self.fc(x)  # [batch_size, num_classes]
        return output
    
class CustomEmbeddingLayer(nn.Module):
    def __init__(self, original_embeddings):
        super(CustomEmbeddingLayer, self).__init__()
        self.original_embeddings = original_embeddings
        self.position_embeddings = original_embeddings.position_embeddings
        self.token_type_embeddings = original_embeddings.token_type_embeddings
        self.LayerNorm = original_embeddings.LayerNorm
        self.dropout = original_embeddings.dropout

    def forward(
        self, input_ids=None, inputs_embeds=None, position_ids=None, token_type_ids=None, past_key_values_length=0
    ):
        if inputs_embeds is None:
            raise ValueError("`inputs_embeds` must be provided for the custom embedding layer.")

        embeddings = inputs_embeds  # [batch_size, 1, 768]

        if position_ids is None:
            # for length == 1
            position_ids = torch.zeros(inputs_embeds.size(0), 1, dtype=torch.long, device=inputs_embeds.device)

        position_embeddings = self.position_embeddings(position_ids)  # [batch_size, 1, 768]
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(position_ids, dtype=torch.long)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)  # [batch_size, 1, 768]
        embeddings = embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    
def train(model, data_loader, optimizer, scheduler, device):
    i = 1
    model.train()
    for batch in data_loader:
        if(i%10==0):
            print("batch i", i)
        i+=1
        optimizer.zero_grad()
        pixel_values = batch['img_vec'].to(device)
        labels = batch['label'].to(device)

        outputs = model(pixel_values)
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
            pixel_values = batch['img_vec'].to(device)
            labels = batch['label'].to(device)
            outputs = model(pixel_values)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return classification_report(actual_labels, predictions, output_dict=True)


bert_model_name="bert-base-uncased"
num_classes = 2
batch_size = 32
num_epochs = 4
learning_rate = 2e-5

print("start loading")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("start training")


num_fold = 5
Report = [[None] * num_epochs for _ in range(num_fold)]

kf = KFold(n_splits=num_fold, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(image_features)):
    print(f"Fold {fold + 1}")
    
    model = CVBERT(bert_model_name, num_classes).to(device)
    
    #model = BeitForImageClassification.from_pretrained(beit_model_name, num_labels=num_classes, ignore_mismatched_sizes=True).to(device)
    
    # split data
    train_image_paths = [image_features[i] for i in train_idx]
    val_image_paths = [image_features[i] for i in val_idx]
    train_labels = [image_labels[i] for i in train_idx]
    val_labels = [image_labels[i] for i in val_idx]

    train_dataset_fold = BeitDataset(train_image_paths, train_labels)
    val_dataset_fold = BeitDataset(val_image_paths, val_labels)

    train_dataloader = DataLoader(train_dataset_fold, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset_fold, batch_size=batch_size, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    # prepare for the model
    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} (Fold {fold+1})")
        train(model, train_dataloader, optimizer, scheduler, device)
        report = evaluate(model, val_dataloader, device)
        print(report)
        print("\n")
        Report[fold][epoch] = report
        
        
filename = "Contrast_BERT_Pre.txt"

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