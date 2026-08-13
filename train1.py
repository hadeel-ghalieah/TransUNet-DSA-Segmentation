import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.data_loader import BraTS2020Dataset
from models.transunet import SimpleTransUNet


# ============================================================
# Configuration
# ============================================================

DATA_DIR = "/content/drive/MyDrive/final everything/data/data/preprocessed_brats2020"
SAVE_PATH = "./checkpoints"
RESULTS_PATH = "./results"

os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
NUM_CLASSES = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {DEVICE}")


# ============================================================
# Dataset and DataLoader
# ============================================================

dataset = BraTS2020Dataset(DATA_DIR)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(f"Number of training samples: {len(dataset)}")


# ============================================================
# Model, Loss Function and Optimizer
# ============================================================

model = SimpleTransUNet(
    in_channels=4,
    out_channels=NUM_CLASSES
).to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Metrics
# ============================================================

def compute_metrics(outputs, masks, num_classes):
    """
    Compute pixel/voxel accuracy and mean Dice score.
    """

    with torch.no_grad():

        preds = torch.argmax(outputs, dim=1)

        # Accuracy
        correct = (preds == masks).float()
        accuracy = correct.mean()

        # Dice score
        dice_scores = []

        for cls in range(num_classes):

            pred_cls = (preds == cls).float()
            mask_cls = (masks == cls).float()

            intersection = (pred_cls * mask_cls).sum()
            union = pred_cls.sum() + mask_cls.sum()

            dice = (2.0 * intersection) / (union + 1e-8)

            dice_scores.append(dice.item())

        mean_dice = sum(dice_scores) / len(dice_scores)

        return accuracy.item(), mean_dice


# ============================================================
# Training History
# ============================================================

stepwise_losses_transunet = []

epoch_losses_transunet = []
epoch_accuracies_transunet = []
epoch_dice_scores_transunet = []

epoch_times = []
gpu_memory_allocated = []
gpu_memory_reserved = []


# ============================================================
# Training Loop
# ============================================================

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0.0
    total_accuracy = 0.0
    total_dice = 0.0

    start_time = time.time()

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch + 1}/{EPOCHS}"
    )

    for images, masks in progress_bar:

        # Move data to device
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Ensure valid class labels
        masks = torch.clamp(
            masks.long(),
            0,
            NUM_CLASSES - 1
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(images)

        loss = criterion(outputs, masks)

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Record loss
        # ----------------------------------------------------

        loss_value = loss.item()

        stepwise_losses_transunet.append(loss_value)

        epoch_loss += loss_value

        # ----------------------------------------------------
        # Compute metrics
        # ----------------------------------------------------

        accuracy, dice = compute_metrics(
            outputs,
            masks,
            NUM_CLASSES
        )

        total_accuracy += accuracy
        total_dice += dice

        # Update progress bar
        progress_bar.set_postfix(
            loss=f"{loss_value:.4f}",
            dice=f"{dice:.4f}"
        )

    # ========================================================
    # Epoch Statistics
    # ========================================================

    avg_loss = epoch_loss / len(train_loader)

    avg_accuracy = total_accuracy / len(train_loader)

    avg_dice = total_dice / len(train_loader)

    elapsed_time = time.time() - start_time

    # --------------------------------------------------------
    # Store epoch-level metrics
    # --------------------------------------------------------

    epoch_losses_transunet.append(avg_loss)

    epoch_accuracies_transunet.append(avg_accuracy)

    epoch_dice_scores_transunet.append(avg_dice)

    epoch_times.append(elapsed_time)

    # --------------------------------------------------------
    # GPU memory statistics
    # --------------------------------------------------------

    if torch.cuda.is_available():

        memory_allocated = (
            torch.cuda.memory_allocated(DEVICE)
            / (1024 ** 2)
        )

        memory_reserved = (
            torch.cuda.memory_reserved(DEVICE)
            / (1024 ** 2)
        )

    else:

        memory_allocated = 0.0
        memory_reserved = 0.0

    gpu_memory_allocated.append(memory_allocated)

    gpu_memory_reserved.append(memory_reserved)

    # ========================================================
    # Print Epoch Results
    # ========================================================

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
        f" | Loss: {avg_loss:.4f}"
        f" | Accuracy: {avg_accuracy:.4f}"
        f" | Dice: {avg_dice:.4f}"
    )

    print(
        f"Time: {elapsed_time:.2f}s"
        f" | GPU Memory Allocated: {memory_allocated:.2f} MB"
        f" | GPU Memory Reserved: {memory_reserved:.2f} MB"
    )

    # ========================================================
    # Save Checkpoint
    # ========================================================

    if (epoch + 1) % 10 == 0 or (epoch + 1) == EPOCHS:

        checkpoint_path = os.path.join(
            SAVE_PATH,
            f"transunet_epoch_{epoch + 1}.pth"
        )

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(
            f"Model checkpoint saved to: {checkpoint_path}"
        )


# ============================================================
# Save Final Complete Checkpoint
# ============================================================

final_checkpoint = {
    "epoch": EPOCHS,

    "model_state_dict": model.state_dict(),

    "optimizer_state_dict": optimizer.state_dict(),

    "loss": epoch_losses_transunet[-1],

    "model_class": "SimpleTransUNet",

    "model_config": {
        "in_channels": 4,
        "out_channels": NUM_CLASSES
    },

    "training_config": {
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE
    }
}

final_checkpoint_path = os.path.join(
    SAVE_PATH,
    f"transunet_epoch_{EPOCHS}_full.pth"
)

torch.save(
    final_checkpoint,
    final_checkpoint_path
)

print(
    f"\nFinal checkpoint saved to: {final_checkpoint_path}"
)


# ============================================================
# Save Training History
# ============================================================

training_results = {

    "model": "SimpleTransUNet",

    "epochs": EPOCHS,

    "stepwise_loss": np.array(
        stepwise_losses_transunet
    ),

    "metrics": {

        "epoch": list(range(1, EPOCHS + 1)),

        "loss": epoch_losses_transunet,

        "accuracy": epoch_accuracies_transunet,

        "dice": epoch_dice_scores_transunet,

        "epoch_time_seconds": epoch_times,

        "gpu_memory_allocated_mb":
            gpu_memory_allocated,

        "gpu_memory_reserved_mb":
            gpu_memory_reserved
    },

    "config": {

        "in_channels": 4,

        "out_channels": NUM_CLASSES,

        "batch_size": BATCH_SIZE,

        "learning_rate": LEARNING_RATE,

        "epochs": EPOCHS
    }
}


results_path = os.path.join(
    RESULTS_PATH,
    "transunet_final_results.pt"
)

torch.save(
    training_results,
    results_path
)

print(
    f"Training results saved to: {results_path}"
)


# ============================================================
# Save Stepwise Loss Separately
# ============================================================

stepwise_loss_path = os.path.join(
    RESULTS_PATH,
    "stepwise_losses_transunet.npy"
)

np.save(
    stepwise_loss_path,
    np.array(stepwise_losses_transunet)
)

print(
    f"Stepwise losses saved to: {stepwise_loss_path}"
)


print("\nTraining completed successfully.")
