# ECG_SSL_LCD

# Introduction
Code implementation of **Lead Correlation and Decorrelation (LCD)**, a new self-supervised learning (SSL) method for multilead electrocardiograms (ECGs).

W. Liu, S. Pan, S. Chang, Q. Huang, N. Jiang, **Self-Supervised Learning for Electrocardiogram Classification Using Lead Correlation and Decorrelation**, _Applied Soft Computing_, 2025, 172: 112871.

# Requirements
```
numpy==1.22.4
scikit_learn==1.3.0
scipy==1.10.1
torch==1.12.0
torchvision==0.12.0
tqdm==4.61.2
```
# Data
- The Ningbo First Hospital (NFH), PTB-XL, CPSC, and Chapman database can be downloaded from:
  - [https://physionet.org/content/challenge-2021/1.0.3/training/ningbo/#files-panel](https://physionet.org/content/challenge-2021/1.0.3/training/ningbo/#files-panel)
  - [https://physionet.org/content/ptb-xl/1.0.3/](https://physionet.org/content/ptb-xl/1.0.3/)
  - [http://2018.icbeb.org/Challenge.html](http://2018.icbeb.org/Challenge.html)
  - [https://figshare.com/collections/ChapmanECG/4560497/2](https://figshare.com/collections/ChapmanECG/4560497/2)
- Records with multiple labels are removed. 
- Each ECG should be transferred to a numpy array (shape: C(lead)×L(length)) and saved as an npy file (.npy). The resampling is implemented by a function from SciPy (scipy.signal.resample); Z-score normalizes each ECG.
- The dataset for pretraining should be structured as follows:
```bash
pt_data_dir/
├── samples
    ├── sample_0.npy
    ├── sample_1.npy
    └── ...
```

- The dataset for a downstream task should be structured as follows:
```bash
down_data_dir/
├── class_x
│   ├── sample_0.npy
│   ├── sample_1.npy
│   └── ...
└── class_y
    ├── sample_0.ext
    ├── sample_1.ext
    └── ...
```
- Please refer to the paper for more information. 
# Pretraining

- run_pt.py
```bash
usage: run_pt.py [-h] --data-dir DIR [--workers N] [--epochs N]
                 [--batch-size N] [--learning-rate LR] [--gamma L]
```

- data-dir is required.
```bash
python run_pt.py --data-dir pt_data_dir
```
# Linear Probing

- run_lp.py
```bash
usage: run_lp.py [-h] --data-dir DIR --checkpoint DIR --num-classes N
                 [--feat-dir DIR] [--workers N] [--epochs N] [--batch-size N]
                 [--learning-rate LR]
```

- data-dir, checkpoint, num-classes are required.
```bash
python run_lp.py --data-dir down_data_dir --checkpoint checkpoint_file.pth --num-classes N
```
# Fine-Tuning

- run.ft.py
```bash
usage: run_ft.py [-h] --data-dir DIR --checkpoint DIR --num-classes N
                 [--fraction L] [--model-dir DIR] [--workers N] [--epochs N]
                 [--batch-size N] [--learning-rate LR]
```

- data-dir, checkpoint, num-classes are required.
```bash
python run_ft.py --data-dir down_data_dir --checkpoint checkpoint_file.pth --num-classes N
```


**WARNING: This code implementation does not support multi-gpu training!**

