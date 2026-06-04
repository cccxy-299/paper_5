# Paper_3 Repository

This repository contains the source code and experiment settings for **Feature-Shift-Aware Framework for Traffic Accident Detection in Surveillance Videos**. All identifying information has been removed to comply with the double-blind review policy.

# 1. Environment Setup

Create a clean environment:
- conda create -n FSA python=3.7
- conda activate FSA
- pip install -r requirements.txt
# 2. Dataset
  SO-TAD dataset [SO-TAD: A Surveillance-Oriented Benchmark for Traffic Accident Detection](https://github.com/cccxy-299/so-tad)
   
   UCF dataset[Real-world anomaly detection in surveillance videos](https://opendatalab.org.cn/OpenDataLab/UCF-Crime)
# 3. Training
   **For So-tad**:python main.py --log_dir "log_path" --dataset_root "path to SO-TAD"
   
   **For UCF**:python main.py --log_dir "log_path"  --tad_root "path to UCF"

