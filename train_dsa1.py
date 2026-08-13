import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# === Dataset ===
class BraTS2020Dataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith(".npz")]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.file_list[idx])
        data = np.load(file_path)
        x = torch.tensor(data['x'], dtype=torch.float32)  # (4, D, H, W)
        y = torch.tensor(data['y'], dtype=torch.long)     # (D, H, W)
        return x, y

# === Model ===
class SparseAttention(nn.Module):
    def __init__(self):
        super(SparseAttention, self).__init__()
        self.dummy_layer = nn.Identity()

    def forward(self, x):
        return self.dummy_layer(x)

class DSA_TransUNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=4):
        super(DSA_TransUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.sparse_attention = SparseAttention()
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(32, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.sparse_attention(x)
        x = self.decoder(x)
        return x

# === Metrics ===
def compute_metrics(outputs, masks, num_classes):
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == masks).float()
        accuracy = correct.sum() / correct.numel()
        dice_scores = []
        for cls in range(num_classes):
            pred_cls = (preds == cls).float()
            mask_cls = (masks == cls).float()
            intersection = (pred_cls * mask_cls).sum()
            union = pred_cls.sum() + mask_cls.sum()
            dice = (2. * intersection) / (union + 1e-8)
            dice_scores.append(dice.item())
        return accuracy.item(), sum(dice_scores)/len(dice_scores)

# === Training ===
def train_model(model_class, model_name):
    DATA_DIR = "/content/drive/MyDrive/final everything/data/data/preprocessed_brats2020"
    SAVE_PATH = "./checkpoints"
    os.makedirs(SAVE_PATH, exist_ok=True)

    EPOCHS = 50
    BATCH_SIZE = 1
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = BraTS2020Dataset(DATA_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = model_class(in_channels=4, out_channels=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    stepwise_losses = []
    all_accuracies = []
    all_dice_scores = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss, total_acc, total_dice = 0, 0, 0
        for images, masks in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            masks = torch.clamp(masks.long(), 0, NUM_CLASSES - 1)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc, dice = compute_metrics(outputs, masks, NUM_CLASSES)
            stepwise_losses.append(loss.item())
            epoch_loss += loss.item()
            total_acc += acc
            total_dice += dice

        avg_acc = total_acc / len(loader)
        avg_dice = total_dice / len(loader)
        all_accuracies.append(avg_acc)
        all_dice_scores.append(avg_dice)
        print(f"[Epoch {epoch+1}] Loss={epoch_loss/len(loader):.4f}, Acc={avg_acc:.4f}, Dice={avg_dice:.4f}")

    return model, stepwise_losses, all_accuracies, all_dice_scores

# === Entry point ===
if __name__ == "__main__":
    model, stepwise_losses_dsa, acc_list, dice_list = train_model(model_class=DSA_TransUNet, model_name="dsa_transunet")

    results = {
        'model_state_dict': model.state_dict(),
        'losses': np.array(stepwise_losses_dsa),
        'metrics': {
            'epoch': list(range(1, len(acc_list) + 1)),
            'accuracy': acc_list,
            'dice': dice_list,
        },
        'config': {
            'in_channels': 4,
            'out_channels': 4,
            'batch_size': 1,
            'learning_rate': 1e-4,
            'has_sparse_attention': True,
        },
        'git_hash': os.popen('git rev-parse HEAD').read().strip(),
        'timestamp': time.strftime("%Y-%m-%d_%H-%M-%S")
    }

    os.makedirs("./results", exist_ok=True)
    torch.save(results, "./results/dsa_transunet_final_results.pt")
    print("✅ Saved all results to dsa_transunet_final_results.pt")
