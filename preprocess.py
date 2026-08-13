import os
import nibabel as nib
import numpy as np
from tqdm import tqdm
from skimage.transform import resize
import matplotlib.pyplot as plt

# Set your BraTS2020 data path
DATASET_PATH = "/home/shitty-programmers/Downloads/archive (1)/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"

# Output directory for preprocessed .npz files
OUTPUT_DIR = "./preprocessed_brats2020"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Desired shape (3D volume: D, H, W)
IMG_SHAPE = (128, 128, 128)

def normalize(img):
    img = np.clip(img, np.percentile(img, 1), np.percentile(img, 99))
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    return img

def load_patient(folder_path):
    modalities = ['t1', 't1ce', 't2', 'flair']
    images = []

    # Load and normalize each modality
    for mod in modalities:
        file = [f for f in os.listdir(folder_path) if mod in f.lower()][0]
        img = nib.load(os.path.join(folder_path, file)).get_fdata()
        img = normalize(img)
        img = resize(img, IMG_SHAPE, mode='constant', preserve_range=True)
        images.append(img)

    # Load and resize segmentation
    seg_file = [f for f in os.listdir(folder_path) if 'seg' in f.lower()][0]
    seg = nib.load(os.path.join(folder_path, seg_file)).get_fdata()
    seg = resize(seg, IMG_SHAPE, order=0, preserve_range=True, anti_aliasing=False)
    seg = seg.astype(np.uint8)

    # Map BraTS labels: 0=background, 1=necrosis, 2=edema, 4=enhancing tumor
    seg[seg == 4] = 3  # Map 4 → 3, so all labels are in [0, 1, 2, 3]

    return np.stack(images, axis=0), seg  # X shape: (4, D, H, W), Y shape: (D, H, W)

# Process all patients
patients = [p for p in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, p))]
print(f"Found {len(patients)} patients.")

for p in tqdm(patients):
    folder_path = os.path.join(DATASET_PATH, p)
    try:
        x, y = load_patient(folder_path)
        np.savez_compressed(os.path.join(OUTPUT_DIR, f"{p}.npz"), x=x, y=y)
    except Exception as e:
        print(f"Error processing {p}: {e}")
