import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BeitImageProcessor, get_linear_schedule_with_warmup
from transformers import BeitModel
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report
from PIL import Image

root_dir = "origin/Dog_Cat"
image_names = [f for f in os.listdir(root_dir) if f.lower().endswith('.jpg')]
image_paths = []
labels = []
for name in image_names:
    path = os.path.join(root_dir, name)
    if "cat." in name:
        label = 0
    elif "dog." in name:
        label = 1
    else:
        # if there are images which are named other than cat/dog
        continue
    image_paths.append(path)
    labels.append(label)
    
# Image Processor which is like Tokeninzer in BERT
image_processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224-pt22k-ft22k")


class BeitDataset(Dataset):
    def __init__(self, image_paths, labels, image_processor):
        self.image_paths = image_paths
        self.labels = labels
        self.image_processor = image_processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        encoding = self.image_processor(img, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        return {
            "pixel_values": pixel_values,
            "label": torch.tensor(label)
        }
        
class BEiTClassifier(nn.Module):
    def __init__(self, beit_model_name, num_classes):
        super(BEiTClassifier, self).__init__()
        self.beit = BeitModel.from_pretrained(beit_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.beit.config.hidden_size, num_classes)

    def forward(self, pixel_values):
        outputs = self.beit(pixel_values=pixel_values)
        pooled_output = outputs.last_hidden_state[:, 0, :]  # Extract CLS token
        pooled_output = pooled_output.contiguous()  # Ensure contiguous memory layout
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits
    
def train(model, data_loader, optimizer, scheduler, device):
    i = 1
    model.train()
    for batch in data_loader:
        if(i%10==0):
            print("batch i", i)
        i+=1
        optimizer.zero_grad()
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['label'].to(device)

        outputs = model(pixel_values=pixel_values)
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
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['label'].to(device)
            outputs = model(pixel_values=pixel_values)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return classification_report(actual_labels, predictions, output_dict=True)

beit_model_name = "microsoft/beit-base-patch16-224-pt22k-ft22k"
num_classes = 2
batch_size = 32
num_epochs = 4
learning_rate = 2e-5

#device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_fold = 5
Report = [[None] * num_epochs for _ in range(num_fold)]

kf = KFold(n_splits=num_fold, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
    print(f"Fold {fold + 1}")
    
    model = BEiTClassifier(beit_model_name, num_classes).to(device)
    
    
    # split data
    train_image_paths = [image_paths[i] for i in train_idx]
    val_image_paths = [image_paths[i] for i in val_idx]
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]

    train_dataset_fold = BeitDataset(train_image_paths, train_labels, image_processor)
    val_dataset_fold = BeitDataset(val_image_paths, val_labels, image_processor)

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
        
filename = "Result_Baseline_High_BEiT.txt"

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