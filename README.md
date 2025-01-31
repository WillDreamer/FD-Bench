# Fluid Dynamics Benchmark (FD-Bench)

## Overview

FD-Bench is a framework for benchmarking and training fluid dynamics models. It includes prebuilt datasets, model architectures, and utilities designed for fluid dynamics-related tasks. The repository is structured to support ease of use and scalability for both researchers and practitioners.

---

## Requirements

Dependencies are listed in **requirements.txt**. You can install them using the following command:

```bash
pip install -r requirements.txt
```

---

## Installation

To install FD-Bench in editable mode:

```bash
pip install -e .
```

---

## Directory Structure

```
FD-Bench/
├── setup.py               # Installation setup script
├── ckpt/                  # Pretrained checkpoints and logs
│   ├── checkpoint_afno_last.pth
│   └── log.txt
├── runs/                  # Output of model runs (e.g., logs, checkpoints)
├── src/                   # Scripts for training and experimentation
│   ├── train.py           # Training script
│   ├── train.sh           # Bash script for training automation
│   ├── engine.py          # Training and evaluation engine
│   ├── train-argpase.py   # Argument parsing for training configuration
├── config/                # Model and training configuration files
│   └── self_atten+linear+var.yaml
├── tree.py                # Script to generate directory tree
├── fdbench/               # Core library
│   ├── models/            # Model architectures
│   │   ├── self_atten/    # Self-attention-based models
│   │   │   ├── afno2d.py
│   │   │   ├── sa.py
│   │   │   ├── gfn.py
│   │   │   ├── afno1d.py
│   │   │   ├── ls.py
│   │   │   ├── self_atten.py
│   │   │   └── bfno2d.py
│   │   └── graph/         # Graph-based models (future support)
│   ├── data/              # Dataset utilities and samplers
│   │   ├── samplers.py
│   │   ├── datasets.py
│   │   └── CNS_data_utils.py
│   └── utils/             # General utilities
│       ├── losses.py      # Loss functions
│       └── utils.py       # General-purpose utilities
├── LICENSE                # License information
├── README.md              # Project documentation
└── fdbench.egg-info/      # Package metadata
```

---

## Usage

### Training

You can start training by providing the following arguments in `src/train.sh`:
- `SPATIAL_REP`: Spatial representation (choices include *graph*, *fourier*, *self-atten*).
- `TEMPORAL_REP`: Temporal representation (choices include *next_step*, *n-ode*).
- `TARGET`: Target variable or field.

Run the following command to train a model:

```bash
bash src/train.sh
```

### Configurations

Model configurations are stored in the `config/` directory. You can modify `self_atten+linear+var.yaml` to adjust model hyperparameters, architecture, or training settings.

---

## Checkpoints and Logs

- Pretrained checkpoints are stored in the `ckpt/` directory. Use these checkpoints to resume training or evaluate pre-trained models.
- Training logs and outputs are saved in the `runs/` directory.

---

## Contributing

Feel free to open issues or submit pull requests for improvements and bug fixes. Contributions are welcome!

---

