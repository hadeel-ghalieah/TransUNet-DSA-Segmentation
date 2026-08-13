# TransUNet-DSA-Segmentation

## TransUNet with Directional/Spatial Attention (DSA) for 3D Medical Image Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch\&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Acceleration-76B900?logo=nvidia\&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

A research-oriented implementation of **TransUNet integrated with Directional/Spatial Attention (DSA)** for **3D brain tumor segmentation** on the **BraTS2020** dataset.

The project investigates the integration of spatial attention into a Transformer-based medical image segmentation architecture, with the goal of improving the model's ability to capture spatially important features and delineate tumor boundaries.

---

## 📌 Project Overview

| Category                  | Details                                         |
| ------------------------- | ----------------------------------------------- |
| **Domain**                | Computer Vision / Medical Image Analysis        |
| **Task**                  | 3D Brain Tumor Segmentation                     |
| **Dataset**               | BraTS2020                                       |
| **Baseline**              | TransUNet                                       |
| **Proposed Method**       | TransUNet + Directional/Spatial Attention (DSA) |
| **Framework**             | PyTorch                                         |
| **Language**              | Python                                          |
| **Hardware Acceleration** | CUDA / GPU                                      |

---

## 🧠 Methodology

The proposed approach extends the **TransUNet** architecture by integrating a **Directional/Spatial Attention (DSA)** mechanism into the segmentation pipeline.

### Architecture

```text
3D MRI Volume
      │
      ▼
Preprocessing
      │
      ▼
TransUNet Encoder
      │
      ▼
Transformer-based Feature Extraction
      │
      ▼
Directional/Spatial Attention (DSA)
      │
      ▼
Decoder
      │
      ▼
3D Tumor Segmentation
```

### Key Idea

**TransUNet** combines convolutional feature extraction with Transformer-based global context modeling.

In this project, **Directional/Spatial Attention (DSA)** is incorporated to enhance the model's focus on spatially informative regions and improve the representation of tumor boundaries.

The proposed architecture is evaluated against the baseline **TransUNet** on the **BraTS2020** dataset.

---

## 📊 Experimental Results

The repository includes experiments comparing the baseline **TransUNet** with the proposed **TransUNet + DSA** approach.

### Stepwise Loss Comparison

The following plot compares the training behavior of the baseline and DSA-enhanced models across training steps.

![Stepwise Loss Comparison](results/stepwise_loss_comparison.png)

### Model Comparison

![TransUNet vs DSA Comparison](results/updated_transunet_vs_dsa_comparison.png)

> The reported experiments focus on comparing training behavior and model performance between the baseline TransUNet architecture and the proposed DSA-enhanced architecture.

---

## 🗂️ Repository Structure

```text
TransUNet-DSA-Segmentation/
│
├── models/
│   └── Model architecture and attention components
│
├── utils/
│   └── Supporting utilities
│
├── results/
│   ├── stepwise_loss_comparison.png
│   └── updated_transunet_vs_dsa_comparison.png
│
├── preprocess.py
├── train1.py
├── train_dsa1.py
├── plot_stepwise_loss_comparison.py
│
└── README.md
```

---

## ⚙️ Technologies

* **Python**
* **PyTorch**
* **CUDA**
* **NumPy**
* **Computer Vision**
* **Medical Image Processing**
* **Transformer-based Deep Learning**
* **3D Image Segmentation**

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/hadeel-ghalieah/TransUNet-DSA-Segmentation.git
cd TransUNet-DSA-Segmentation
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Linux/macOS:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

Install the required Python packages for the project.

```bash
pip install torch torchvision numpy scipy opencv-python matplotlib
```

> For GPU acceleration, install the appropriate PyTorch/CUDA configuration for your system.

### 4. Prepare the dataset

Download and prepare the **BraTS2020** dataset according to its official data access requirements.

Then configure the dataset paths in the preprocessing/training scripts before running the experiments.

### 5. Preprocess the data

```bash
python preprocess.py
```

### 6. Train the baseline TransUNet

```bash
python train1.py
```

### 7. Train the TransUNet + DSA model

```bash
python train_dsa1.py
```

### 8. Generate loss comparisons

```bash
python plot_stepwise_loss_comparison.py
```

---

## 🔬 Research Focus

This project explores the use of attention mechanisms within Transformer-based architectures for **3D medical image segmentation**.

The main research questions include:

* How does integrating DSA affect TransUNet's segmentation behavior?
* Does spatial attention improve the representation of tumor regions?
* How does the modified architecture compare with the baseline TransUNet?
* What are the computational and memory implications of the proposed modification?

---

## 🎯 Applications

The techniques explored in this project are relevant to:

* Medical image segmentation
* Brain tumor analysis
* Computer-aided diagnosis
* 3D MRI analysis
* Computer Vision
* Transformer-based medical AI

---

## 📚 Dataset

Experiments are conducted using the **BraTS2020** brain tumor segmentation dataset.

BraTS provides multimodal MRI scans together with expert annotations for brain tumor segmentation and is widely used for evaluating automated brain tumor segmentation methods.

Please follow the dataset's official terms and access requirements when obtaining and using the data.

---

## 👩‍💻 Author

**Hadeel Ghalieah**

MSc Computer Science Engineering — University of Pécs
Focus: **Machine Learning, Computer Vision & Deep Learning**

* GitHub: https://github.com/hadeel-ghalieah
* LinkedIn: https://linkedin.com/in/hadeel-ghalieah
* Email: [hadeel.ghalieah@gmail.com](mailto:hadeel.ghalieah@gmail.com)

---

## 📄 License

This project is released under the **MIT License**.
