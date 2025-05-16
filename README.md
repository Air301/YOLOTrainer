# YOLOTrainer

## 项目概述

YOLOTrainer 是一个用于训练、评估和部署 YOLO (You Only Look Once) 家族目标检测模型的工具集。该项目旨在简化 YOLO 模型的训练流程，并适配主流的开源遥感目标检测数据集，使研究人员和开发者能够更高效地训练自定义目标检测模型。

## 功能特点

### 已完成开发

- 支持 YOLOv11 模型的训练和微调
- 基础数据集预处理工具
- 模型导出工具
- 基础推理功能

### 开发中功能

- 支持 YOLOe 的训练和微调
- YOLOe 预训练模型辅助标注工具

## 安装指南

### 前提条件

- Python 3.8+
- CUDA 兼容的 GPU（推荐用于训练）
- Git

### 安装步骤

#### 方法一：使用 pip 和 venv

```bash
# 克隆仓库
git clone https://github.com/yourusername/YOLOTrainer.git
cd YOLOTrainer

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # 在 Windows 上使用 venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

#### 方法二：使用 Poetry

```bash
# 克隆仓库
git clone https://github.com/yourusername/YOLOTrainer.git
cd YOLOTrainer

# 安装 Poetry（如果尚未安装）
curl -sSL https://install.python-poetry.org | python3 -

# 使用 Poetry 安装依赖
poetry install

# 激活虚拟环境
poetry shell
```

#### 方法三：使用 Conda

```bash
# 克隆仓库
git clone https://github.com/yourusername/YOLOTrainer.git
cd YOLOTrainer

# 创建 Conda 环境
conda create -n yolotrainer python=3.8
conda activate yolotrainer

# 安装依赖
pip install -r requirements.txt

# 或者使用 conda 安装依赖（如果有 environment.yml）
# conda env create -f environment.yml
```