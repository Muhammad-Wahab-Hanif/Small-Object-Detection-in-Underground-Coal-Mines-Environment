# MFF-YOLOv5s-4EL: Multi-Feature Fusion YOLOv5s with 4 Enhancement Layers

**Underground Coal Mine Small Object Detection**

[![Python 3.8](https://img.shields.io/badge/Python-3.8-blue.svg)](https://www.python.org/)
[![PyTorch 1.9.0](https://img.shields.io/badge/PyTorch-1.9.0-red.svg)](https://pytorch.org/)
[![CUDA 11.1](https://img.shields.io/badge/CUDA-11.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Abstract

This repository implements **MFF-YOLOv5s-4EL** (Multi-Feature Fusion YOLOv5s with 4 Enhancement Layers) for small object detection in underground coal mine environments. Our model integrates five key improvements: SIoU loss, Decoupled Detection Head, Small Object Detection Layer (P2 feature map), SimAM attention mechanism, and C3Ghost lightweight modules.

## 🎯 Performance

| Model | Params (M) | GFLOPs | Size (MB) | mAP (%) |
|-------|-----------|--------|-----------|---------|
| YOLOv5s | 21.7 | 15.9 | 14.4 | 92.6 |
| MFF-YOLOv5s-4EL | 25.0 | 16.0 | 17.2 | **97.9** |

## 📦 Detected Classes

- `person` - Underground miners
- `stone` - Rock/stone obstacles
- `coal` - Coal deposits
- `signal_light` - Signal light indicator
- `left_light` - Turn left signal light
- `right_light` - Turn right signal light
- `turnout` - Locomotive track turnout/switch
- `electric_locomotive` - Electric locomotive vehicle

## 🖥️ Hardware Requirements

| Component | Specification |
|-----------|---------------|
| OS | Ubuntu 18.04 |
| CPU | Intel Xeon Platinum 8350C @ 2.6 GHz |
| GPU | NVIDIA RTX 3090 (24 GiB) |
| RAM | 32GB+ recommended |
| Storage | 100GB+ for dataset and checkpoints |

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/MFF-YOLOv5s-4EL.git
cd MFF-YOLOv5s-4EL

# Create conda environment (recommended)
conda create -n mff_yolo python=3.8
conda activate mff_yolo

# Install dependencies
pip install -r requirements.txt