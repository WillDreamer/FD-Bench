# A Fair Benchmark for AI-powered Fluid Dynamics Modeling (FD-Bench)

## 🧭  Overview

FD-Bench is the most fair and comprehensive framework for benchmarking and training fluid dynamics models, tracing hundreds of papers and decomposing all of the methods into modules. It includes prebuilt datasets, model architectures, and utilities designed for fluid dynamics-related tasks. The repository is structured to support ease of use and scalability for both researchers and practitioners.

---

## 💻  Requirements

Dependencies are listed in **requirements.txt**. You can install them using the following command:

```bash
conda create -n fdbench python=3.10
pip install -r requirements.txt
```

All our experiments are based on 8 $\times$ A6000 (48G).

---

## 🔧  Installation

To install FD-Bench in editable mode:

```bash
git clone https://github.com/xxxxx/FD-Bench.git
cd FD-Bench
pip install -e .
```

---

## 📊  Data Preparation

Our benchmark is validated on various types of PDE from multiple data sources.
- For `CNS` (compressible N-S Equation) data, we follow the setting of [**PDEBench**](https://github.com/pdebench/PDEBench).
- For `DR` (diffusion-reaction Equation) data, we follow the setting of [**PDEBench**](https://github.com/pdebench/PDEBench).
- For `KF` (incompressible N-S Equation ) data, we follow the setting of [**Posiden**](https://huggingface.co/collections/camlab-ethz/poseidon-downstream-tasks-664fa237cd6b0c097971ef14).


## 🤗  Hugging Face Datasets

We provide several fluid dynamics datasets hosted on [🤗 Hugging Face Datasets]. To load them:

```bash
pip install datasets
```

Then in your script:

```python
from datasets import load_dataset

# Example: load the Advection2 dataset
ds = load_dataset("xxxxxx/FD-Bench-Advection2")
```

No Hugging Face token or authentication is required to access these datasets.

---

## 📖  Directory Structure

```
FD-Bench/
├── setup.py               # Installation setup script
├── exps/                  # Pretrained checkpoints and logs
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
│   │   │── graph/         # Graph-based models 
│   │   └── diffusion/         # denoising-based models 
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

## 🧑‍💻  Usage

### 🧪  Training the model

1. You can start training by providing the following arguments in `src/train.sh`:
- `SPATIAL_REP`: Spatial representation (choices include *graph*, *fourier*, *self-atten* and so on).
- `TEMPORAL_REP`: Temporal representation (choices include *next_step*, *n-ode* and so on).
- `TARGET`: Prediction target of the model (choices include *physical variable*, *random noise* *flow field*.

2. Model configurations are stored in the `config/` directory. You can modify `TARGET/SPATIAL_REP+TEMPORAL_REP.yaml` to adjust model hyperparameters, architecture, or training settings.

3. Run the following command to train a model:

```bash
bash src/train.sh
```
The default setting starts with distributed training with multiple GPUs. Note that NCCL Backend does not support ComplexFloat data type, so avoid using DDP when `SPATIAL_REP='Fourier'`


### 📖 Modular Design

- For `TARGET = noise`, you will be took to `FD-Bench/config/noise`, and we don't specialize the `SPATIAL_REP` in the name. You can refer to the args inside for changing saptial architecture.

- For `TARGET = PDE Residual`, there is no specific yaml. You need to add `if_pde_residual: True` and `if_coordinate: True` in the yaml file. Then it will find the corresponding residual loss in `fdbench/utils/pde_utils.py`. For example, add these two args in `self_atten+next_step.yaml`, it will add the PDE residual loss.



### 🔝  Checkpoints and Logs

- Pretrained checkpoints are stored in the `ckpt/` directory. Use these checkpoints to resume training or evaluate pre-trained models.
- Training logs and outputs are saved in the `runs/` directory.
- Wandb logs are saved in the `wandb/` directory (optional). You need to claim the `WANDB_ENTITY`, `WANDB_PROJECT` and `WANDB_API_KEY` in the `train.sh`.
```bash
export WANDB_ENTITY="xxx"
export WANDB_PROJECT="xxx"
export WANDB_NAME="xxx"
export WANDB_API_KEY="xxx"
```

---

## 🔥  Contributing

### 🚀  Model Extension
If you add your own model within this benchmark framework, you should:
- Add a folder under `models` with the name **your_module**.
- Create `module.py` under `FD-Bench/fdbench/models/your_module`
- Output the `output` and `loss` within the *forward* function in your model class.

### 🏠  (Optional) Build from Public Library
For example, if you want to use the model from public library such as `neuraloperator`, you can:

1. Install the package
```bash

git submodule add https://github.com/neuraloperator/neuraloperator.git fdbench/models/neuraloperator
git submodule update --init --recursive

cd fdbench/models/neuraloperator
pip install -e .
# pip install -r requirements.txt
```

2. Define the config file in `config/public` and use the corresponding parameters to initialize `model` in `src/train.py`.
   
3. Run the code
   
```bash

bash src/train_public.sh
```

### 🤝  Collaboration
Feel free to open issues or submit pull requests for improvements and bug fixes. Contributions are welcome! Drop me a line at `xxxx'📮📮📮

---

