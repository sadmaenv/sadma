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

### distributed

For starting a distributed experiment, there are two steps. The first step is to start the train server, and the second step is to start the sample worker.

Start train server

- edit `configs/config_utils.py`. Set async_train to True and set role to train. Set args.address of train,sample equal to train server ip address. Param num_sample_worker means the number of all sample workers at all machine. You might want to setup other params here.

- run it just via: python entry.py

If you do not want to change the role,num_sample_worker at file, you can do it this way: python entry.py --role train --num_sample_worker 16

Start sample worker. 

- edit `configs/config_utils.py`. Set role to sample. And then setup num_sample_worker and sampler_id, it depends on how many samper machine you have. The meaning of num_sample_worker here is different from server, it means the number of sample workers at current machine. If you only got two machines, one is train server and the other is sample worker. In that case, you just keep num_sample_worker as before set at server, and set sampler_id to 0. If you got more than two machine and you want to use more machine to sample. You need to decide how many sample workers to put on every sample machine.

- run it just via: python entry.py. Then go back to set up num_sample_worker and sampler_id if you got more sample machine.

For example, set num_sample_worker to 16 when start train server. And you have 4 sample machine, you can put 4 sample workers on every machine. In this case, you need to set  num_sample_worker to 4 when start sample worker and set sampler_id to 0,4,8,and 12 sequentially

    ```shell
    python entry.py --role train --num_sample_worker 16
    python entry.py --role sample --num_sample_worker 4
    python entry.py --role sample --num_sample_worker 4 --sampler_id 4
    python entry.py --role sample --num_sample_worker 4 --sampler_id 8
    python entry.py --role sample --num_sample_worker 4 --sampler_id 12
    ```
