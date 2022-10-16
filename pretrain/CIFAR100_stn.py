import os

import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR100

from datas.STN import Alignment, FreezeSTN


def run_stn(yaml):
    Stn_Translate_List = ["ShearX", "ShearY", "TranslateX", "TranslateY", "Rotate"]
    for index in range(len(Stn_Translate_List)):
        if not os.path.exists(
            os.path.join(yaml["SDA"]["pretrain_path"], f"{Stn_Translate_List[index]}.pth")
        ):
            system = Alignment(
                Stn_Translate_List[index],
                [32, 32] if yaml["SDA"]["dataset_type"] == "CIFAR" else [224, 224],
                os.path.join(yaml["SDA"]["pretrain_path"], f"{Stn_Translate_List[index]}.pth"),
                FreezeSTN,
                dataset_type=yaml["SDA"]["dataset_type"],
            )
            trainset = torchvision.datasets.CIFAR100(
                root=yaml["data_path"],
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
                    ]
                ),
            )

            train_dataloader = DataLoader(
                trainset, shuffle=True, num_workers=4, batch_size=64, pin_memory=True
            )
            system.fit(train_dataloader)
