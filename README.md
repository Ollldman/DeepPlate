# DeepPlate: Multimodal Deep Learning for Calorie Estimation (educational project)

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-FFD21E.svg?logo=huggingface&logoColor=black)
![Albumentations](https://img.shields.io/badge/Albumentations-008080.svg?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTEyLjUgMTRsLTQuNS00LjVhLjc1Ljc1IDAgMDExLjA2LTEuMDZMNy41IDEwbC0uNS41YS43NS43NSAwIDAxLTEuMDYgMCAuNzUuNzUgMCAwMS0uMDYtLjA2TDEyLjUgMy41bDcuNSA3LjVhLjc1Ljc1IDAgMDEtLjA2IDEuMDZsLS41LjVhLjc1Ljc1IDAgMDEtMS4wNiAwTDkuNSAxNGwtNC41IDQuNWEuNzUuNzUgMCAwMS0xLjA2IDBMLDMuNWEuNzUuNzUgMCAwMS4wNi0xLjA2TDEyLjUgMTAuNWw3LjUtNy41YS43NS43NSAwIDEwIDEuMDZhLjc1Ljc1IDAgMDEwIDEuMDZMOS41IDE0eiIvPjwvc3ZnPg==)
![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?logo=nvidia&logoColor=white)
![Poetry](https://img.shields.io/badge/Poetry-2B72C9?logo=poetry&logoColor=white)


This project implements a multimodal deep learning model to predict the calorie content of meals using both **food images** and **textual ingredient descriptions**. The model leverages pretrained vision (e.g., ResNet, MobileNetV3) and language (e.g., BERT) backbones, with a fusion mechanism and partial fine-tuning to achieve accurate calorie regression.

Target metric: **Mean Absolute Error (MAE) < 50** on the test set.

## Key Features

- 📸 **Multimodal input**: RGB food images + ingredient text  
- 🧠 **Fine-tuning** of pretrained vision and language encoders  
- 🔗 **Custom fusion architecture** for cross-modal interaction (e.g., cross-attention)  
- 🧪 **Reproducible training pipeline** with config-based setup and seed control  
- ⚡ **GPU-accelerated training and evaluation** (CUDA support)

## Repository Structure
.

├── data/                   # Symlink or path to dataset (not committed)

├── src/

│   ├── dataset.py          # Multimodal dataset & dataloaders

│   ├── model.py            # Multimodal model definition

│   └── utils.py            # Training, validation, and reproducibility utilities

├── notebook.ipynb          # EDA, training, and inference

└── README.md

## Requirements

- Python 3.13+
- PyTorch ≥ 2.0
- torchvision
- [🤗 Transformers](https://huggingface.co/docs/transformers)
- [Albumentations](https) (for advanced image augmentations)
- pandas, numpy, matplotlib, etc.

Install dependencies via:
```bash
poetry install
```

> Note: Actual dataset (~1.3 GB) is expected to be placed in `data/` following the provided structure.

## Goal

Select and train a multimodal model suitable for integration into health and fitness applications that helps users estimate meal calories from a photo and ingredient list.
