import argparse
import os
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from omegaconf import OmegaConf

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


if __name__ == "__main__":
    yaml_config = yaml_config_get(args)
    device = set_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.fastest = True
    set_random_seed(0)
    wandb.init(project="RLDCD", entity="seushanshan")
    if yaml_config["amp"]:
        scaler = torch.cuda.amp.GradScaler()
    tnet: nn.Module = getattr(models, yaml_config["tarch"])(
        num_classes=yaml_config["num_classes"]
    ).cuda()
    net: nn.Module = getattr(models, yaml_config["arch"])(
        num_classes=yaml_config["num_classes"]
    ).cuda()
    ROOT = yaml_config["ckpt_root"]
    local_ckpt_path = yaml_config["local_ckpt_path"]
    if yaml_config["tcheckpoint"]:
        tcheckpoint_path = ROOT + yaml_config["tcheckpoint"]
        tnet = utils.load_model_from_url(tnet, tcheckpoint_path, local_ckpt_path)
    else:
        raise NotImplementedError("the teacher2's checkpoint file could not be found!")
    optimizer_name = "torch.optim." + yaml_config["optimizer"]["type"]
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
        wandb=wandb,
    )
    env.training_in_all_epoch()
