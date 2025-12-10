# train.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import FiberContourDataset
from model import PointNetLike
from tqdm import tqdm

# Setting
DATA_DIR = "data"
NUM_POINTS = 64
BATCH_SIZE = 32
EPOCHS = 100
LR = 0.001  
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_PATH = "pointnet_pore_classifier.pth"

def main():
    # dataset
    train_dataset = FiberContourDataset(DATA_DIR, num_points=NUM_POINTS, split='train')
    val_dataset = FiberContourDataset(DATA_DIR, num_points=NUM_POINTS, split='val')
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Model
    model = PointNetLike(num_classes=2, num_points=NUM_POINTS).to(DEVICE) # 2 classes
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # train
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for points, scalars, labels in progress_bar:
            points, scalars, labels = points.to(DEVICE), scalars.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(points, scalars)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for points, scalars, labels in val_loader:
                points, scalars, labels = points.to(DEVICE), scalars.to(DEVICE), labels.to(DEVICE)
                logits = model(points, scalars)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} finished. Avg Loss: {avg_loss:.4f}, Val Acc: {acc:.4f}")
    
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()