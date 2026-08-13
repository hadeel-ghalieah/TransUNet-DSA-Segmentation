import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time
import os
import numpy  as np
from utils.data_loader import BraTS2020Dataset
from models.transunet import SimpleTransUNet

# === Config ===
DATA_DIR = "/content/drive/MyDrive/final everything/data/data/preprocessed_brats2020"
SAVE_PATH = "./checkpoints"
os.makedirs(SAVE_PATH, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Dataset and Loader ===
dataset = BraTS2020Dataset(DATA_DIR)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model, Loss, Optimizer ===
model = SimpleTransUNet(in_channels=4, out_channels=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def compute_metrics(outputs, masks, num_classes):
    with torch.no_grad():
        preds = torch.argmax(outputs, dim=1)  # (B, D, H, W)
        correct = (preds == masks).float()
        accuracy = correct.sum() / correct.numel()

        # Dice score
        dice_scores = []
        for cls in range(num_classes):
            pred_cls = (preds == cls).float()
            mask_cls = (masks == cls).float()
            intersection = (pred_cls * mask_cls).sum()
            union = pred_cls.sum() + mask_cls.sum()
            dice = (2. * intersection) / (union + 1e-8)
            dice_scores.append(dice.item())

        avg_dice = sum(dice_scores) / len(dice_scores)
        return accuracy.item(), avg_dice


# Your model, criterion, optimizer, etc. must be defined earlier in the script
stepwise_losses_transunet = []

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    total_accuracy = 0.0
    total_dice = 0.0
    start_time = time.time()

    for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)
        masks = torch.clamp(masks.long(), 0, NUM_CLASSES - 1)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        stepwise_losses_transunet.append(loss_value)
        epoch_loss += loss_value

        acc, dice = compute_metrics(outputs, masks, NUM_CLASSES)
        total_accuracy += acc
        total_dice += dice

    end_time = time.time()
    elapsed_time = end_time - start_time

    avg_accuracy = total_accuracy / len(train_loader)
    avg_dice = total_dice / len(train_loader)

    memory_allocated = torch.cuda.memory_allocated(DEVICE) / (1024 ** 2) if torch.cuda.is_available() else 0
    memory_reserved = torch.cuda.memory_reserved(DEVICE) / (1024 ** 2) if torch.cuda.is_available() else 0

    print(f"Epoch {epoch+1} - Loss: {epoch_loss / len(train_loader):.4f} | Accuracy: {avg_accuracy:.4f} | Dice: {avg_dice:.4f}")
    print(f"Time: {elapsed_time:.2f}s | GPU Memory Allocated: {memory_allocated:.2f}MB | Reserved: {memory_reserved:.2f}MB")

    if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:
        checkpoint_path = os.path.join(SAVE_PATH, f"transunet_epoch_{epoch+1}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to: {checkpoint_path}")
#my adding
# At the end of training (after the last epoch):
checkpoint = {
    'epoch': EPOCHS,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss / len(train_loader),
    'model_class': 'SimpleTransUNet',  # Explicitly state the class
    'model_config': {                  # Save architecture details
        'in_channels': 4,
        'out_channels': 4,
    },
    'git_hash': os.popen('git rev-parse HEAD').read().strip()  # Optional: Link to code version
}
torch.save(checkpoint, f"./checkpoints/transunet_epoch_{EPOCHS}_full.pth")
# Save stepwise loss
np.save("stepwise_losses_transunet.npy", np.array(stepwise_losses_transunet))
print("Saved stepwise losses to stepwise_losses_transunet.npy")
# At the end of training (after the last epoch), add this:
results = {
    'model_state_dict': model.state_dict(),
    'losses': np.array(stepwise_losses_transunet),  # Stepwise losses
    'metrics': {  # Example: Track per-epoch metrics
        'epoch': list(range(1, EPOCHS + 1)),
        'accuracy': [...],  # Populate with your recorded accuracies
        'dice': [...],     # Populate with your recorded Dice scores
    },
    'config': {  # Hyperparameters and model info
        'in_channels': 4,
        'out_channels': 4,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
    },
    'git_hash': os.popen('git rev-parse HEAD').read().strip(),  # Link to code version
    'timestamp': time.strftime("%Y-%m-%d_%H-%M-%S")  # When the run finished
}

# Save everything to a single file
torch.save(results, "./results/transunet_final_results.pt")
print("✅ Saved all results to transunet_final_results.pt")
