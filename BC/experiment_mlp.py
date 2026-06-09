import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

# Configurations
TRAIN_PATH = r"d:\Kaggle_final\data\2p\train.npz"
VAL_PATH = r"d:\Kaggle_final\data\2p\val.npz"
TEST_PATH = r"d:\Kaggle_final\data\2p\test.npz"
BATCH_SIZE = 4096
EPOCHS = 20
LEARNING_RATE = 1e-3
PATIENCE = 3

class BehaviorCloningDataset(Dataset):
    def __init__(self, npz_path):
        data = np.load(npz_path)
        self.X = torch.tensor(data['X'], dtype=torch.float32)
        self.y = torch.tensor(data['y'], dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FlexibleMLP(nn.Module):
    def __init__(self, input_dim=46, hidden_dims=[64, 32]):
        super().__init__()
        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_d, h))
            layers.append(nn.ReLU())
            in_d = h
        layers.append(nn.Linear(in_d, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x)

def train_and_evaluate(model_name, hidden_dims, train_loader, val_loader, test_loader, device):
    print(f"\n{'='*60}\nTraining Architecture: {model_name} {hidden_dims}\n{'='*60}")
    model = FlexibleMLP(hidden_dims=hidden_dims).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    patience_counter = 0
    save_path = f"d:/Kaggle_final/best_{model_name.lower()}.pth"
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': []
    }
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        all_train_preds, all_train_labels = [], []
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            all_train_preds.append((outputs > 0.5).cpu().numpy())
            all_train_labels.append(batch_y.cpu().numpy())
            
        train_loss /= len(train_loader.dataset)
        train_preds = np.vstack(all_train_preds)
        train_labels = np.vstack(all_train_labels)
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, zero_division=0)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds, all_val_labels = [], []
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                all_val_preds.append((outputs > 0.5).cpu().numpy())
                all_val_labels.append(batch_y.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_preds = np.vstack(all_val_preds)
        val_labels = np.vstack(all_val_labels)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        print(f"Epoch {epoch+1:02d} | Train [L: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}] | Val [L: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}]")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping triggered at Epoch {epoch+1}.")
                break
                
    # Evaluate Test set using the best model
    model.load_state_dict(torch.load(save_path))
    model.eval()
    all_test_preds, all_test_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            all_test_preds.append((outputs > 0.5).cpu().numpy())
            all_test_labels.append(batch_y.cpu().numpy())
    test_preds = np.vstack(all_test_preds)
    test_labels = np.vstack(all_test_labels)
    test_acc = accuracy_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, test_preds, zero_division=0)
    print(f"\n>>> FINAL TEST RESULTS FOR '{model_name}' - Accuracy: {test_acc:.4f} | F1-Score: {test_f1:.4f}")
    print(f"    (Model saved to {save_path})")
    
    return history

def plot_metrics(history, save_path):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(18, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss', marker='o', color='red')
    plt.plot(epochs, history['val_loss'], label='Val Loss', marker='o', color='blue')
    plt.title('Loss over Epochs (Shallow MLP)')
    plt.xlabel('Epochs')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    
    # Accuracy plot
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['train_acc'], label='Train Accuracy', marker='o', color='red')
    plt.plot(epochs, history['val_acc'], label='Val Accuracy', marker='o', color='blue')
    plt.title('Accuracy over Epochs (Shallow MLP)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    
    # F1 Score plot
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['train_f1'], label='Train F1-Score', marker='o', color='red')
    plt.plot(epochs, history['val_f1'], label='Val F1-Score', marker='o', color='blue')
    plt.title('F1-Score over Epochs (Shallow MLP)')
    plt.xlabel('Epochs')
    plt.ylabel('F1-Score')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot successfully saved to: {save_path}")

def run_experiments():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting experiments. Using device: {device}")
    
    print("Loading datasets... (This may take a few moments)")
    train_dataset = BehaviorCloningDataset(TRAIN_PATH)
    val_dataset = BehaviorCloningDataset(VAL_PATH)
    test_dataset = BehaviorCloningDataset(TEST_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    architectures = {
        'Shallow': [64, 32],
        'Medium': [128, 64],
        'Deep': [256, 128, 64]
    }
    
    for name, dims in architectures.items():
        history = train_and_evaluate(name, dims, train_loader, val_loader, test_loader, device)
        
        # Draw plot only for Shallow model as requested
        if name == 'Shallow':
            plot_metrics(history, r"d:\Kaggle_final\shallow_mlp_metrics.png")

if __name__ == "__main__":
    run_experiments()
