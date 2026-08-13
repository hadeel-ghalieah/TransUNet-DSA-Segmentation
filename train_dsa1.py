import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.dsa_attention import DSA_TransUNet


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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {DEVICE}")


# ============================================================
# Dataset
# ============================================================

class BraTS2020Dataset(Dataset):

    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.file_list = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith(".npz")
        ])

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):

        file_path = os.path.join(
            self.data_dir,
            self.file_list[idx]
        )

        data = np.load(file_path)

        x = torch.tensor(
            data["x"],
            dtype=torch.float32
        )

        y = torch.tensor(
            data["y"],
            dtype=torch.long
        )

        return x, y


# ============================================================
# Metrics
# ============================================================

def compute_metrics(outputs, masks, num_classes):

    with torch.no_grad():

        preds = torch.argmax(
            outputs,
            dim=1
        )

        # Accuracy
        correct = (preds == masks).float()

        accuracy = correct.mean()

        # Dice
        dice_scores = []

        for cls in range(num_classes):

            pred_cls = (preds == cls).float()

            mask_cls = (masks == cls).float()

            intersection = (
                pred_cls * mask_cls
            ).sum()

            union = (
                pred_cls.sum()
                + mask_cls.sum()
            )

            dice = (
                2.0 * intersection
            ) / (union + 1e-8)

            dice_scores.append(
                dice.item()
            )

        mean_dice = (
            sum(dice_scores)
            / len(dice_scores)
        )

        return accuracy.item(), mean_dice


# ============================================================
# Dataset and DataLoader
# ============================================================

dataset = BraTS2020Dataset(DATA_DIR)

train_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

print(
    f"Number of training samples: {len(dataset)}"
)


# ============================================================
# Model
# ============================================================

model = DSA_TransUNet(
    in_channels=4,
    out_channels=NUM_CLASSES
).to(DEVICE)

print("\nModel:")
print(model)


# ============================================================
# Loss and Optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training History
# ============================================================

stepwise_losses = []

epoch_losses = []
epoch_accuracies = []
epoch_dice_scores = []
epoch_times = []


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

        images = images.to(DEVICE)

        masks = masks.to(DEVICE)

        masks = torch.clamp(
            masks.long(),
            0,
            NUM_CLASSES - 1
        )

        # ----------------------------------------------------
        # Forward pass
        # ----------------------------------------------------

        outputs = model(images)

        loss = criterion(
            outputs,
            masks
        )

        # ----------------------------------------------------
        # Backward pass
        # ----------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        loss_value = loss.item()

        accuracy, dice = compute_metrics(
            outputs,
            masks,
            NUM_CLASSES
        )

        stepwise_losses.append(
            loss_value
        )

        epoch_loss += loss_value

        total_accuracy += accuracy

        total_dice += dice

        progress_bar.set_postfix(
            loss=f"{loss_value:.4f}",
            dice=f"{dice:.4f}"
        )

    # ========================================================
    # Epoch Results
    # ========================================================

    avg_loss = (
        epoch_loss
        / len(train_loader)
    )

    avg_accuracy = (
        total_accuracy
        / len(train_loader)
    )

    avg_dice = (
        total_dice
        / len(train_loader)
    )

    elapsed_time = (
        time.time() - start_time
    )

    epoch_losses.append(avg_loss)

    epoch_accuracies.append(
        avg_accuracy
    )

    epoch_dice_scores.append(
        avg_dice
    )

    epoch_times.append(
        elapsed_time
    )

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
        f" | Loss: {avg_loss:.4f}"
        f" | Accuracy: {avg_accuracy:.4f}"
        f" | Dice: {avg_dice:.4f}"
        f" | Time: {elapsed_time:.2f}s"
    )

    # ========================================================
    # Save Checkpoint
    # ========================================================

    if (
        (epoch + 1) % 10 == 0
        or (epoch + 1) == EPOCHS
    ):

        checkpoint_path = os.path.join(
            SAVE_PATH,
            f"dsa_transunet_epoch_{epoch + 1}.pth"
        )

        torch.save(
            model.state_dict(),
            checkpoint_path
        )

        print(
            f"Checkpoint saved: {checkpoint_path}"
        )


# ============================================================
# Save Final Model + Results
# ============================================================

results = {

    "model_name":
        "DSA_TransUNet",

    "model_state_dict":
        model.state_dict(),

    "losses":
        np.array(stepwise_losses),

    "metrics": {

        "epoch":
            list(range(1, EPOCHS + 1)),

        "loss":
            epoch_losses,

        "accuracy":
            epoch_accuracies,

        "dice":
            epoch_dice_scores,

        "epoch_time_seconds":
            epoch_times
    },

    "config": {

        "in_channels":
            4,

        "out_channels":
            NUM_CLASSES,

        "batch_size":
            BATCH_SIZE,

        "learning_rate":
            LEARNING_RATE,

        "epochs":
            EPOCHS,

        "attention": {

            "type":
                "Windowed Multi-Head Self-Attention",

            "heads":
                4,

            "dim_head":
                32,

            "window_size":
                5
        }
    },

    "timestamp":
        time.strftime(
            "%Y-%m-%d_%H-%M-%S"
        )
}


results_file = os.path.join(
    RESULTS_PATH,
    "dsa_transunet_final_results.pt"
)

torch.save(
    results,
    results_file
)


# ============================================================
# Save Stepwise Loss
# ============================================================

loss_file = os.path.join(
    RESULTS_PATH,
    "stepwise_losses_dsa_transunet.npy"
)

np.save(
    loss_file,
    np.array(stepwise_losses)
)


print("\n========================================")
print("Training completed successfully.")
print(f"Results saved to: {results_file}")
print(f"Stepwise loss saved to: {loss_file}")
print("========================================")
