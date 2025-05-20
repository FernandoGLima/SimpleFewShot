
![Simple Few Shot](assets/logo.png)

[![PyPI](https://img.shields.io/pypi/v/simplefsl?label=simplefsl&logo=python&style=for-the-badge)](https://pypi.org/project/simplefsl)
[![Colab](https://img.shields.io/badge/Open%20In-Colab-FFD43B?logo=googlecolab&logoColor=black&style=for-the-badge)](https://colab.research.google.com/drive/11m4Dbvgpnm4HVaGsuaXE-mjwuaPY30Ik?usp=sharing)


## Overview

A modular library for **few-shot learning**, implementing classic and state-of-the-art meta-learning algorithms

It's designed to be flexible, extensible, and easy to integrate into your research or production pipeline

- Plug-and-play support for multiple few-shot learning models
- Episodic training and evaluation setup
- Clean dataset interface for custom datasets
- Support for custom backbones like **ResNet**, **ViT**, and more
- Built-in data augmentation options (`basic`, `mixup`, `cutmix`)

## Available Models

| Model           | Paper Reference |
|-----------------|-----------------|
| **ProtoNet**    | [https://arxiv.org/abs/1703.05175](https://arxiv.org/abs/1703.05175) |
| **RelationNet** | [https://arxiv.org/abs/1711.06025](https://arxiv.org/abs/1711.06025) |
| **MatchingNet** | [https://arxiv.org/abs/1606.04080](https://arxiv.org/abs/1606.04080) |
| **MetaOptNet**  | [https://arxiv.org/abs/1904.03758](https://arxiv.org/abs/1904.03758) |
| **TapNet**      | [https://arxiv.org/abs/1905.06549](https://arxiv.org/abs/1905.06549) |
| **TADAM**       | [https://arxiv.org/abs/1805.10123](https://arxiv.org/abs/1805.10123) |
| **DN4**         | [https://arxiv.org/abs/1903.12290](https://arxiv.org/abs/1903.12290) |
| **MSENet**      | [https://arxiv.org/abs/2409.07989v2](https://arxiv.org/abs/2409.07989v2) |
| **FEAT**        | [https://arxiv.org/abs/1812.03664](https://arxiv.org/abs/1812.03664) |

All models are implemented as separate `.py` files under the [models/](/simplefsl/models) directory

## Getting Started

Install the package directly from GitHub:

```bash
pip install git+https://github.com/victor-nasc/SimpleFewShot.git
```

A ready-to-run Colab example is available:

[![Colab](https://img.shields.io/badge/Open%20In-Colab-FFD43B?logo=googlecolab&logoColor=black&style=for-the-badge)](https://colab.research.google.com/drive/11m4Dbvgpnm4HVaGsuaXE-mjwuaPY30Ik?usp=sharing)

## Custom Dataset Format

To use a custom dataset, provide a `.csv` file with the following structure:

- `image_id`: Path to the image file
- One column per class (binary: 1 if image belongs to class, 0 otherwise)

**Example:**

| idx | image_id         | class1 | class2 | class3 |
|-----|------------------|--------|--------|--------|
| 1   | `/data/img1.jpg` | 1      | 0      | 0      | 
| 2   | `/data/img2.jpg` | 0      | 1      | 0      | 
| 3   | `/data/img3.jpg` | 0      | 0      | 1      |
| ... | ...              | ...    | ...    | ...    | 

You can define your training and test class splits programmatically using the `FewShotManager` class

Obs.: This method supports also **multi-labeled** datasets

## Citation

If you find this library useful in your research or project, please consider citing:
```
...
```
