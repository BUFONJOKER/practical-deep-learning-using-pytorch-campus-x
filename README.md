
---

# Practical Deep Learning using PyTorch-CampusX Course

## Course Introduction 👋
This repository contains the complete, hands-on notebook series for the Practical Deep Learning using PyTorch course by CampusX. The materials guide you from PyTorch fundamentals to full training pipelines, GPU acceleration, CNNs, RNNs/LSTMs, and hyperparameter tuning. Each notebook is self-contained and designed for step-by-step learning.

## Table of Contents 🧭
- [What You’ll Learn](#what-youll-learn-)
- [Prerequisites](#prerequisites-)
- [Setup](#setup-)
- [How to Use](#how-to-use-)
- [Notebooks Overview](#notebooks-overview-)
- [Project Structure](#project-structure-)
- [Contributing](#contributing-)
- [License](#license-)

## What You’ll Learn ✅
- PyTorch tensors, autograd, and gradient mechanics
- Training loops, optimization, and evaluation
- Modular model design with `nn.Module`
- Data pipelines with `Dataset` and `DataLoader`
- GPU training and performance optimization
- CNNs for image classification (MNIST)
- RNNs/LSTMs for sequence modeling
- Hyperparameter tuning with Optuna

## Prerequisites 📌
- Basic Python programming
- Familiarity with linear algebra and calculus is helpful
- Recommended: prior exposure to machine learning basics

## Setup ⚙️
1. Clone the repository.
2. Create a Python environment (venv/conda).
3. Install dependencies such as PyTorch, torchvision, and Jupyter.
4. Launch Jupyter and open any notebook.

## How to Use 🧑‍💻
- Start from notebook 01 and proceed in order.
- Each notebook includes explanations and runnable code.
- For GPU notebooks, ensure CUDA-enabled PyTorch is installed.

## Notebooks Overview 📚

### 01. Autograd Foundations 🧮
- File: [01-autograd.ipynb](01-autograd.ipynb)
- Details: Core tensor operations, automatic differentiation concepts, and gradient tracking basics.

### 02. Training Pipeline Basics 🏗️
- File: [02-training-pipeline.ipynb](02-training-pipeline.ipynb)
- Details: End-to-end training loop structure, loss computation, and parameter updates.

### 03. PyTorch `nn.Module` Essentials 🧩
- File: [03-pytorch-nn-module.ipynb](03-pytorch-nn-module.ipynb)
- Details: Building models with `nn.Module`, defining forward passes, and parameter management.

### 04. Training Pipeline with `nn.Module` 🧪
- File: [04-pytorch-training-pipeline-using-nn-module.ipynb](04-pytorch-training-pipeline-using-nn-module.ipynb)
- Details: Modular training pipeline using `nn.Module`, optimizers, and evaluation steps.

### 05. Dataset & DataLoader Fundamentals 📦
- File: [05-dataset-dataloader.ipynb](05-dataset-dataloader.ipynb)
- Details: Custom datasets, batching with DataLoader, shuffling, and data preprocessing.

### 06. Training with Dataset & DataLoader 🚚
- File: [06-pytorch-training-pipeline-using-nn-module-using-dataset-dataloader.ipynb](06-pytorch-training-pipeline-using-nn-module-using-dataset-dataloader.ipynb)
- Details: Full training loop integrated with datasets and dataloaders for scalable training.

### 07. MNIST ANN Baseline 🧠
- File: [07-mnist-data-ann.ipynb](07-mnist-data-ann.ipynb)
- Details: Feedforward ANN on MNIST, baseline accuracy, and training diagnostics.

### 08. MNIST ANN on GPU ⚡
- File: [08-mnist-data-ann-usin-gpu.ipynb](08-mnist-data-ann-usin-gpu.ipynb)
- Details: GPU acceleration for ANN training and performance comparison.

### 09. MNIST ANN GPU Optimizations 🚀
- File: [09-mnist-data-ann-using-gpu-optimized.ipynb](09-mnist-data-ann-using-gpu-optimized.ipynb)
- Details: Optimized training settings, improved throughput, and stability tweaks.

### 10. MNIST CNN Full Pipeline 🖼️
- File: [10-mnist-dataset-full-cnn-using-gpu-optimized.ipynb](10-mnist-dataset-full-cnn-using-gpu-optimized.ipynb)
- Details: Full CNN model on MNIST with GPU optimization and evaluation.

### 11. MNIST CNN Transfer Learning 🔁
- File: [11-mnist-data-cnn-transfers-learning.ipynb](11-mnist-data-cnn-transfers-learning.ipynb)
- Details: Transfer learning workflow, fine-tuning, and feature reuse concepts.

### 12. RNN Fundamentals 🔄
- File: [12-rnn.ipynb](12-rnn.ipynb)
- Details: Recurrent neural network basics, sequence modeling, and training steps.

### 13. Next-Word Prediction with LSTM ✍️
- File: [13-next-word-predictor-LSTM.ipynb](13-next-word-predictor-LSTM.ipynb)
- Details: LSTM architecture for language modeling and next-word prediction.

### 14. Optuna Hyperparameter Tuning 🎯
- File: [14-optuna-hyperparameter-tuning-framework.ipynb](14-optuna-hyperparameter-tuning-framework.ipynb)
- Details: Automated hyperparameter search using Optuna and evaluation metrics.

## Project Structure 🗂️
- Notebooks are organized numerically in the root folder for linear learning.
- [mlflow_pytorch.py](mlflow_pytorch.py) contains helper code for experiment tracking.

## Contributing 🤝
Contributions are welcome. Please open an issue for suggestions or submit a pull request with clear changes and context.

## License 📄
This project is for educational use. If a specific license is required, add a LICENSE file or update this section.

---