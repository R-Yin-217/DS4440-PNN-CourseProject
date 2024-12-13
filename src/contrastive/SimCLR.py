# 必要なライブラリのインポート
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

# define SimCLR
class SimCLR(nn.Module):
    def __init__(self, base_encoder, projection_dim=768):
        super(SimCLR, self).__init__()
        self.encoder = base_encoder
        dim_mlp = self.encoder.fc.weight.shape[1]
        
        # Projection Head
        self.encoder.fc = nn.Identity() 
        self.projection_head = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, projection_dim)
        )
    
    def forward(self, x):
        h = self.encoder(x)  # extract representations from encoder
        z = self.projection_head(h)  # Projection Head
        return z

# loss function
class NTXentLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5, device='cuda'):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.mask = self._get_correlated_mask().type(torch.bool)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
    
    def _get_correlated_mask(self):
        N = 2 * self.batch_size
        mask = torch.ones((N, N)) - torch.eye(N)
        return mask.to(self.device)
    
    def forward(self, z_i, z_j):
        N = 2 * self.batch_size
        z = torch.cat((z_i, z_j), dim=0)
        z = F.normalize(z, dim=1)
        
        # do cosine similarity
        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        
        # apply mask
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        positives = torch.cat([sim_i_j, sim_j_i], dim=0).view(N, 1)
        negatives = sim[self.mask].view(N, -1)
        
        logits = torch.cat((positives, negatives), dim=1)
        labels = torch.zeros(N).to(self.device).long()
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

# define custom dataset
class SimCLRDataset(datasets.ImageFolder):
    def __init__(self, root, transform):
        super(SimCLRDataset, self).__init__(root, transform)
    
    def __getitem__(self, index):
        try:
            (path, label) = self.samples[index]
            sample = self.loader(path)
            if self.transform is not None:
                xi = self.transform(sample)
                xj = self.transform(sample)
            return xi, xj
        except Exception as e:
            print(f"Error loading image {self.samples[index][0]}: {e}")
            return self.__getitem__((index + 1) % len(self))

# define data agumentation
simclr_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
    ], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# make dataloader for dataset
data_dir = "origin/Dog_Cat"
batch_size = 256  
dataset = SimCLRDataset(root=data_dir, transform=simclr_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_encoder = models.resnet50(pretrained=False)  
simclr_model = SimCLR(base_encoder, projection_dim=786).to(device)

# define loss and optimizer
criterion = NTXentLoss(batch_size=batch_size, temperature=0.5, device=device)
optimizer = optim.Adam(simclr_model.parameters(), lr=1e-4, weight_decay=1e-6) 


scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)  


epochs = 100

# train loop
simclr_model.train()
for epoch in range(epochs):
    total_loss = 0
    for (xi, xj) in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
        xi = xi.to(device)
        xj = xj.to(device)
        
        optimizer.zero_grad()
        
        zi = simclr_model(xi)
        zj = simclr_model(xj)
        
        loss = criterion(zi, zj)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
    scheduler.step()
    
    if((epoch+1)%10==0):
        torch.save(simclr_model.state_dict(), f"simclr_epoch_{epoch+1}.pth")


simclr_model.eval()

# data loader
eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

eval_dataset = datasets.ImageFolder(root=data_dir, transform=eval_transform)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

embeddings = []
file_paths = []

with torch.no_grad():
    for batch_idx, (inputs, _) in enumerate(tqdm(eval_dataloader, desc="Extracting Embeddings")):
        inputs = inputs.to(device)
        z = simclr_model(inputs)
        z = z.cpu().numpy()
        embeddings.append(z)
        
        batch_start = batch_idx * batch_size
        batch_end = batch_start + inputs.size(0)
        file_paths.extend([path for path, _ in eval_dataset.samples[batch_start:batch_end]])


embeddings = np.vstack(embeddings)

# transform into dataframe
df = pd.DataFrame(embeddings)
df['file_path'] = file_paths

# save as csv
df.to_csv("image_embeddings_786d.csv", index=False)
print("Embeddings have been saved to 'image_embeddings_768d.csv'")