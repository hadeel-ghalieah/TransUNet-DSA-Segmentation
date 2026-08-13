# TransUNet-DSA-Segmentation
# TransUNet with Directional/Spatial Attention (DSA) for Medical Image Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

This repository implements **TransUNet integrated with Directional/Spatial Attention (DSA)** for high-precision medical image segmentation. By combining the global context modeling of Transformers with localized spatial attention mechanisms, this model improves boundary delineation and segmentation performance over standard U-Net and baseline TransUNet architectures.

---

## 📌 Project Overview

* **Domain:** Computer Vision / Medical Image Analysis
* **Task:** Semantic Segmentation
* **Core Architecture:** TransUNet + DSA (Directional / Spatial Attention)
* **Framework:** PyTorch

---

## 📊 Experimental Results & Loss Comparison

The incorporation of the DSA mechanism accelerates convergence and lowers training/validation loss across training steps compared to standard TransUNet.

### Stepwise Loss Comparison
![Stepwise Loss Comparison](results/stepwise_loss_comparison.png)

### Model Comparison & Evaluation
![TransUNet vs DSA Comparison](results/updated_transunet_vs_dsa_comparison.png)

---
