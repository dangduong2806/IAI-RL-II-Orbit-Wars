import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import time

# Configurations
TRAIN_PATH = r"d:\Kaggle_final\data\2p\train.npz"
VAL_PATH = r"d:\Kaggle_final\data\2p\val.npz"
TEST_PATH = r"d:\Kaggle_final\data\2p\test.npz"
MODEL_SAVE_PATH = r"d:\Kaggle_final\mlp_commander.pth"
BATCH_SIZE = 4096
EPOCHS = 20
LEARNING_RATE = 1e-3
PATIENCE = 3 # Number of epochs with no improvement after which training will be stopped

class BehaviorCloningDataset(Dataset):
    def __init__(self, npz_path):
        print(f"Loading data from {npz_path}...")
        data = np.load(npz_path)
        self.X = torch.tensor(data['X'], dtype=torch.float32)
        self.y = torch.tensor(data['y'], dtype=torch.float32).unsqueeze(1) # shape (N, 1)
        print(f"Loaded {len(self.X)} samples.")
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class MLPCommander(nn.Module):
    def __init__(self, input_dim=46):
        super(MLPCommander, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataset and DataLoader
    train_dataset = BehaviorCloningDataset(TRAIN_PATH)
    val_dataset = BehaviorCloningDataset(VAL_PATH)
    test_dataset = BehaviorCloningDataset(TEST_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = MLPCommander().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        start_time = time.time()
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                
        val_loss /= len(val_loader.dataset)
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_time:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Model saved to {MODEL_SAVE_PATH}")
        else:
            patience_counter += 1
            print(f"  -> Early stopping counter: {patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                print("Early stopping triggered! Halting training.")
                break
            
    print("\nTraining completed! Evaluating on test set...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item() * batch_X.size(0)
            
            # Tính accuracy cho threshold 0.5
            predicted = (outputs > 0.5).float()
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
    test_loss /= len(test_loader.dataset)
    test_acc = correct / total
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:.2f}%")

if __name__ == "__main__":
    train()
