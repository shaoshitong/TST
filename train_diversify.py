import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data.distributed
import torchvision
import torchvision.transforms as transforms
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb
from Env.Environment2 import *

sys.path.append(os.path.join(os.getcwd()))
import datas
import losses
import models
import utils

parser = argparse.ArgumentParser(description="RLDCD archive")
parser.add_argument(
    "--config_file",
    type=str,
    default="./configs/wrn40_2_wrn16_2_c100_diversify.yaml",
    help="path to configuration file",
)
parser.add_argument(
    "--cuda_devices",
    type=str,
    default="0",
    help="data parallel training",
)
parser.add_argument(
    "--resume",
    type=str,
    default="",
    help="resume checkpoint",
)
args = parser.parse_args()


def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


def set_random_seed(number=0):
    torch.manual_seed(number)
    torch.cuda.manual_seed(number)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    import random

    import numpy as np

    np.random.seed(number)
    random.seed(number)


def yaml_config_get(args):
    if args.config_file is None:
        print("No config file provided, use default config file")
        return None
    else:
        print("Config file provided:", args.config_file)
    conf = OmegaConf.load(args.config_file)
    return conf


def main_worker(gpu, yaml_config, ngpus_per_node, world_size, dist_url):
    # TODO: NCCL  INIT
    print("Use GPU: {} for training".format(gpu))
    rank = 0  # 单机
    dist_backend = "nccl"
    rank = rank * ngpus_per_node + gpu
    print("world_size:", world_size)
    dist.init_process_group(
        backend=dist_backend, init_method=dist_url, world_size=world_size, rank=rank
    )

    set_random_seed(rank + np.random.randint(0, 1000))
    torch.cuda.set_device(gpu)
    yaml_config["train_batch_size"] = int(yaml_config["train_batch_size"] / ngpus_per_node)
    yaml_config["test_batch_size"] = int(yaml_config["test_batch_size"] / ngpus_per_node)

    # TODO: LLA_DFD
    if gpu == 0:
        # os.environ["WANDB_API_KEY"] = "625345833d2e13b7e2c695c406cc01311f39bf40"
        # os.environ["WANDB_MODE"] = "offline"
        wandb.init(project="LLA_DFD", entity="seushanshan")
    tnet: nn.Module = torch.nn.DataParallel(
        getattr(models, yaml_config["tarch"])(num_classes=yaml_config["num_classes"]).cuda(gpu),
        device_ids=[gpu],
    )
    net: nn.Module = DDP(
        getattr(models, yaml_config["arch"])(num_classes=yaml_config["num_classes"]).cuda(gpu),
        device_ids=[gpu],
    )
    ROOT = yaml_config["ckpt_root"]
    local_ckpt_path = yaml_config["local_ckpt_path"]
    if yaml_config["tcheckpoint"]:
        tcheckpoint_path = ROOT + yaml_config["tcheckpoint"]
        tnet = utils.load_model_from_url(tnet, tcheckpoint_path, local_ckpt_path)
    else:
        raise NotImplementedError("the teacher2's checkpoint file could not be found!")
    optimizer = getattr(torch.optim, yaml_config["optimizer"]["type"])(
        net.parameters(),
        lr=yaml_config["optimizer"]["lr"],
        weight_decay=yaml_config["optimizer"]["weight_decay"],
        momentum=yaml_config["optimizer"]["momentum"]
        if "momentum" in yaml_config["optimizer"]
        else 0.9,
        nesterov=True,
    )
    scheduler = getattr(torch.optim.lr_scheduler, yaml_config["scheduler"]["type"])(
        optimizer,
        milestones=yaml_config["scheduler"]["milestones"],
        gamma=yaml_config["scheduler"]["gamma"],
    )
    trainloader, testloader = getattr(datas, yaml_config["data"])(
        data_path=yaml_config["data_path"],
        num_worker=yaml_config["num_worker"],
        train_batch_size=yaml_config["train_batch_size"] // 2,
        test_batch_size=yaml_config["test_batch_size"] // 2,

    )
    criticion = getattr(losses, yaml_config["criticion"]["type"])(
        temperature=yaml_config["criticion"]["temperature"]
    )
    if gpu == 0:
        wandb.config = yaml_config
    env = LearnDiversifyEnv(
        dataloader=trainloader,
        testloader=testloader,
        student_model=net,
        teacher_model=tnet,
        scheduler=scheduler,
        optimizer=optimizer,
        loss=criticion,
        yaml=yaml_config,
        wandb=wandb if gpu == 0 else None,
        gpu=gpu,
    )
    env.training_in_all_epoch()
    dist.destroy_process_group()


if __name__ == "__main__":
    yaml_config = yaml_config_get(args)
    yaml_config["resume"] = "none" if args.resume == "" else args.resume
    device = set_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.fastest = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices
    os.environ["MASTER_ADDR"] = "127.0.0.1"  #
    os.environ["MASTER_PORT"] = "8888"  #
    world_size = 1
    port_id = 10002 + np.random.randint(0, 1000) + int(args.cuda_devices[0])
    dist_url = "tcp://127.0.0.1:" + str(port_id)
    ngpus_per_node = torch.cuda.device_count()
    world_size = ngpus_per_node * world_size
    print("multiprocessing_distributed")
    torch.multiprocessing.set_start_method("spawn")
    mp.spawn(  # Left 2: softmax weight=1 Right 2: softmax weight=2
        main_worker, nprocs=ngpus_per_node, args=(yaml_config, ngpus_per_node, world_size, dist_url)
    )
