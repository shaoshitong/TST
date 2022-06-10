import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import shutil
import argparse,importlib
from omegaconf import OmegaConf
from RL.Environment import PolicyEnv
import numpy as np
import torchvision
import torchvision.transforms as transforms
import os,sys
import wandb


sys.path.append(os.path.join(os.getcwd()))
import models,losses,datas,utils

parser = argparse.ArgumentParser(description='RLDCD archive')
parser.add_argument('--config_file', type=str, default='./configs/wrn40_2_wrn16_2_c100.yaml',
                    help='path to configuration file')
args = parser.parse_args()
def set_device():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device
def yaml_config_get(args):
    if args.config_file is None:
        print('No config file provided, use default config file')
        return None
    else:
        print('Config file provided:', args.config_file)
    conf = OmegaConf.load(args.config_file)
    return conf



if __name__=="__main__":
    yaml_config=yaml_config_get(args)
    device = set_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.fastest = False
    wandb.init(project="RLDCD", entity="seushanshan")
    if yaml_config['amp']:
        scaler = torch.cuda.amp.GradScaler()
    tnet:nn.Module=getattr(models,yaml_config['tarch'])(num_classes=yaml_config['num_classes']).cuda()
    net:nn.Module=getattr(models,yaml_config['arch'])(num_classes=yaml_config['num_classes']).cuda()
    ROOT=r'https://github.com/shaoshitong/torchdistill/releases/download/v0.3.2/'
    if yaml_config['tcheckpoint']:
        tcheckpoint_path=ROOT+yaml_config['tcheckpoint']
        tnet =utils.load_model_from_url(tnet,tcheckpoint_path)
    else:
        raise NotImplementedError("the teacher's checkpoint file could not be found!")
    optimizer_name="torch.optim."+yaml_config['optimizer']['type']
    optimizer=getattr(torch.optim,yaml_config['optimizer']['type'])(net.parameters(),
                                                                    lr=yaml_config['optimizer']['lr'],
                                                                    weight_decay=yaml_config['optimizer']['weight_decay'],
                                                                    momentum=yaml_config['optimizer']['momentum'] if 'momentum' in yaml_config['optimizer'] else 0.9)
    scheduler=getattr(torch.optim.lr_scheduler,yaml_config['scheduler']['type'])(optimizer,milestones=yaml_config['scheduler']['milestones'],gamma=yaml_config['scheduler']['gamma'])
    trainloader,valloader,testloader=getattr(datas,yaml_config['data'])(
        data_path=yaml_config['data_path'],
        val_ratio=yaml_config['val_ratio'],
        num_worker=yaml_config['num_worker'],
        train_batch_size=yaml_config['train_batch_size']//2,
        test_batch_size=yaml_config['test_batch_size']//2,
    )

    criticion=getattr(losses,yaml_config['criticion']['type'])(
        temperature=yaml_config['criticion']['temperature']
    )
    wandb.config=yaml_config
    env=PolicyEnv(
        dataloader=trainloader,
        valloader=valloader,
        testloader=testloader,
        student_model=net,
        teacher_model=tnet,
        scheduler=scheduler,
        optimizer=optimizer,
        loss=criticion,
        yaml=yaml_config,
        wandb=wandb)
    env.training_in_all_epoch()
