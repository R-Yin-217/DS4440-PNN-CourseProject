import os
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BeitImageProcessor,  get_linear_schedule_with_warmup
from transformers import BertModel
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

print("data loaded")

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
        
        
class CVBERT(nn.Module):
    def __init__(self, bert_model_name, img_size=224, patch_size=16, in_chans=3, embed_dim=768, num_classes=2):
        super(CVBERT, self).__init__()
        # patch embedding
        self.patch_embedding = PatchEmbedding(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        
        # load BertModel
        self.bert = BertModel.from_pretrained(bert_model_name)

        # set custom layer
        self.bert.embeddings = CustomEmbeddingLayer(self.bert.embeddings)
        
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

    def forward(self, image, attention_mask=None):
        # 1. put image to the patch embedding
        patch_embeddings = self.patch_embedding(image)  # [batch_size, num_patches, embed_dim]
        batch_size, num_patches, embed_dim = patch_embeddings.size()

        # 2. adjust size of attention size
        if attention_mask is None:
            attention_mask = torch.ones(batch_size, num_patches, device=image.device)

        # 3. put into BERT
        logits = self.bert(
            inputs_embeds=patch_embeddings, attention_mask=attention_mask, input_ids=None
        )
        pooled_output = logits.pooler_output 
        
        
        x = self.dropout(pooled_output)
        output = self.fc(x)
        return output


# Like what ViT did
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # [batch_size, embed_dim, h_patches, w_patches]
        x = x.flatten(2)  # [batch_size, embed_dim, num_patches]
        x = x.transpose(1, 2)  # [batch_size, num_patches, embed_dim]
        return x

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

        embeddings = inputs_embeds

        if position_ids is None:
            # caculate the position embedding based on the patch
            position_ids = torch.arange(
                past_key_values_length, embeddings.size(1) + past_key_values_length,
                dtype=torch.long, device=embeddings.device
            ).unsqueeze(0).expand(embeddings.size()[:2])  # [batch_size, num_patches]

        position_embeddings = self.position_embeddings(position_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(position_ids, dtype=torch.long)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)
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
        pixel_values = batch['pixel_values'].to(device)
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
            pixel_values = batch['pixel_values'].to(device)
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

for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
    print(f"Fold {fold + 1}")
    
    model = CVBERT(bert_model_name, num_classes).to(device)
    
    #model = BeitForImageClassification.from_pretrained(beit_model_name, num_labels=num_classes, ignore_mismatched_sizes=True).to(device)
    
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
        
        
filename = "Result_CV_BERT_Pre.txt"

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