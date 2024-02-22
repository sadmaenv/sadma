# sadma
# Usage

Linux system is recommended.

## Installation instructions

Install Python packages.

```shell
# require Anaconda 3 or Miniconda 3
conda create -n sadma python=3.9 -y
conda activate sadma
pip install -r requirement.txt
```

Install [pytorch](https://pytorch.org)

    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Set up env StarCraft II (2.4.10) and SMAC. More detail could be found at [pymarl2](https://github.com/hijkzzz/pymarl2)

    bash install_sc2.sh

Set up env CityFlow. More detail could be found at [Installation Guide](https://cityflow.readthedocs.io/en/latest/install.html). Build From Source is recommended

    pip install git+https://github.com/cityflow-project/CityFlow.git

Set up env ReplenishmentEnv. More detail could be found at [here](https://github.com/VictorYXL/ReplenishmentEnv)

## Run an experiment 

```
python entry.py --alg qmix --train_device cuda:0 --env_type smac --map_name 3m
```

The config files act as defaults for an algorithm or environment. They are all located in `configs`. More alg config could be found in `configs/alg_configs`. More env config could be found in `configs/env_configs`. If the params are set up right, it is quite east to run an experiment - `python entry.py`.

All results will be stored in the `results` folder( defined in configs/default.yaml log_root_path).
