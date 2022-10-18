import os

import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from datas.COLOR import Alignment, ColorAugmentation


def run_imagenet_color(yaml):
    Color_Translate_List = ["Brightness", "Color", "Contrast", "Sharpness", "Posterize", "Solarize"]
    for index in range(len(Color_Translate_List)):
        if not os.path.exists(
                os.path.join(yaml["SDA"]["pretrain_path"], f"{Color_Translate_List[index]}.pth")
        ):
            system = Alignment(
                Color_Translate_List[index],
                [32, 32] if yaml["SDA"]["dataset_type"] == "CIFAR" else [224, 224],
                os.path.join(yaml["SDA"]["pretrain_path"], f"{Color_Translate_List[index]}.pth"),
                ColorAugmentation,
                dataset_type=yaml["SDA"]["dataset_type"],
                epoch=10,
            )
            trainset = torchvision.datasets.ImageFolder(
                root=yaml["data_path"],

                transform=transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                ),
            )

            from sklearn.model_selection import StratifiedShuffleSplit
            import numpy as np
            import torch
            few_shot_ratio = 0.1
            labels = trainset.targets
            ss = StratifiedShuffleSplit(n_splits=1, test_size=1 - few_shot_ratio, random_state=0)
            train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
            trainset = torch.utils.data.Subset(trainset, train_indices)

            train_dataloader = DataLoader(
                trainset, shuffle=True, num_workers=4, batch_size=64, pin_memory=True
            )
            system.fit(train_dataloader)
