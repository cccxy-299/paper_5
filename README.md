Anonymous Repository for Paper Submission

This repository contains the source code and experiment settings for our paper submitted to an anonymous review venue. All identifying information has been removed to comply with the double-blind review policy.

1. Overview

This repository provides:

Implementation of the proposed model

Training and evaluation scripts

Dataset preprocessing tools

Configuration files for reproducing all experiments

Ablation and analysis setups corresponding to the submitted manuscript

The codebase aims to ensure reproducibility while remaining anonymized.

2. Environment Setup

Create a clean environment:

conda create -n project-env python=3.10
conda activate project-env
pip install -r requirements.txt


Minimal dependency list:

torch>=2.0
numpy
opencv-python
tqdm
einops
pyyaml

3. Repository Structure
.
├── configs/            # Training, evaluation, and ablation configs
├── data/               # Dataset preparation scripts
├── models/             # Model architectures
├── modules/            # Custom blocks and layers
├── utils/              # Common utilities
├── scripts/            # Training and evaluation entry points
├── results/            # Optional logs and sample outputs
└── README.mk
