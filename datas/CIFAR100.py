import os
import numpy as np
import torch
from torch.utils.data import Dataset
import math
import torch
import random
import torchvision.datasets
from torchvision.transforms import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import Dataset




class BaseDatasetWrapper(Dataset):
    def __init__(self, org_dataset):
        self.org_dataset = org_dataset

    def __getitem__(self, index):
        sample, target = self.org_dataset.__getitem__(index)
        return sample, target

    def __len__(self):
        return len(self.org_dataset)


def rotate_with_fill(img, magnitude):
    rot = img.convert('RGBA').rotate(magnitude)
    return Image.composite(rot, Image.new('RGBA', rot.size, (128,) * 4), rot).convert(img.mode)


def shearX(img, magnitude, fillcolor):
    return img.transform(img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0), Image.BICUBIC,
                         fillcolor=fillcolor)


def shearY(img, magnitude, fillcolor):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0), Image.BICUBIC,
                         fillcolor=fillcolor)


def translateX(img, magnitude, fillcolor):
    return img.transform(img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                         fillcolor=fillcolor)


def translateY(img, magnitude, fillcolor):
    return img.transform(img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                         fillcolor=fillcolor)


def rotate(img, magnitude, fillcolor):
    return rotate_with_fill(img, magnitude)


def color(img, magnitude, fillcolor):
    return ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1]))


def posterize(img, magnitude, fillcolor):
    return ImageOps.posterize(img, magnitude)


def solarize(img, magnitude, fillcolor):
    return ImageOps.solarize(img, magnitude)


def contrast(img, magnitude, fillcolor):
    return ImageEnhance.Contrast(img).enhance(1 + magnitude * random.choice([-1, 1]))


def sharpness(img, magnitude, fillcolor):
    return ImageEnhance.Sharpness(img).enhance(1 + magnitude * random.choice([-1, 1]))


def brightness(img, magnitude, fillcolor):
    return ImageEnhance.Brightness(img).enhance(1 + magnitude * random.choice([-1, 1]))


def autocontrast(img, magnitude, fillcolor):
    return ImageOps.autocontrast(img)


def equalize(img, magnitude, fillcolor):
    return ImageOps.equalize(img)


def invert(img, magnitude, fillcolor):
    return ImageOps.invert(img)


class SubPolicy:

    def __init__(self, p1, operation1, magnitude_idx1, fillcolor=(128, 128, 128)):
        self.fillcolor = fillcolor
        ranges = {
            'shearX': np.linspace(0, 0.3, 10),
            'shearY': np.linspace(0, 0.3, 10),
            'translateX': np.linspace(0, 150 / 331, 10),
            'translateY': np.linspace(0, 150 / 331, 10),
            'rotate': np.linspace(0, 30, 10),
            'color': np.linspace(0.0, 0.9, 10),
            'posterize': np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            'solarize': np.linspace(256, 0, 10),
            'contrast': np.linspace(0.0, 0.9, 10),
            'sharpness': np.linspace(0.0, 0.9, 10),
            'brightness': np.linspace(0.0, 0.9, 10),
            'autocontrast': [0] * 10,
            'equalize': [0] * 10,
            'invert': [0] * 10
        }

        func = {
            'shearX': shearX,
            'shearY': shearY,
            'translateX': translateX,
            'translateY': translateY,
            'rotate': rotate,
            'color': color,
            'posterize': posterize,
            'solarize': solarize,
            'contrast': contrast,
            'sharpness': sharpness,
            'brightness': brightness,
            'autocontrast': autocontrast,
            'equalize': equalize,
            'invert': invert
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]

    def __call__(self, img):
        label = 0
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1, self.fillcolor)
            label = 1
        return img, label


class PolicyDatasetC100(BaseDatasetWrapper):
    def __init__(self, org_dataset, p=0.3):
        super(PolicyDatasetC100, self).__init__(org_dataset)
        self.transform = org_dataset.transform
        org_dataset.transform = None
        self.org_dataset=org_dataset
        print("the probability of CIFAR-100 is {}".format(p))
        self.policies = [
            SubPolicy(p, 'autocontrast', 2),
            SubPolicy(p, 'contrast', 3),
            SubPolicy(p, 'posterize', 0),
            SubPolicy(p, 'solarize', 4),

            SubPolicy(p, 'translateY', 8),
            SubPolicy(p, 'shearX', 5),
            SubPolicy(p, 'brightness', 3),
            SubPolicy(p, 'shearY', 0),
            SubPolicy(p, 'translateX', 1),

            SubPolicy(p, 'sharpness', 5),
            SubPolicy(p, 'invert', 4),
            SubPolicy(p, 'color', 4),
            SubPolicy(p, 'equalize', 8),
            SubPolicy(p, 'rotate', 3),

        ]
        self.policies_len = len(self.policies)

    def __getitem__(self, index):
        sample, target = super(PolicyDatasetC100, self).__getitem__(index)
        policy_index = torch.zeros(self.policies_len).float()
        new_sample = sample
        for i in range(self.policies_len):
            new_sample, label = self.policies[i](new_sample)
            policy_index[i] = label
        new_sample = self.transform(new_sample).detach()
        sample = self.transform(sample).detach()
        if isinstance(target, torch.Tensor) and target.ndim == 2 and target.shape[-1] != 1:
            target = target.argmax(1)
        elif not isinstance(target, torch.Tensor):
            target = torch.LongTensor([target])
        target = target.unsqueeze(0).expand(2, -1)  # 2,1
        sample = torch.stack([  # 2,XXX
            sample,
            new_sample,
        ])
        return sample, target


def DataLoader_C100(data_path,val_ratio,num_worker,train_batch_size=64,test_batch_size=64):
    trainset = torchvision.datasets.CIFAR100(root=data_path,train=True, download=True,
                                             transform=transforms.Compose([
                                                 transforms.RandomCrop(32, padding=4),
                                                 transforms.RandomHorizontalFlip(),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                      [0.2675, 0.2565, 0.2761])
                                             ]))
    from sklearn.model_selection import StratifiedShuffleSplit
    labels = [trainset[i][1] for i in range(len(trainset))]
    ss = StratifiedShuffleSplit(n_splits=1, test_size= val_ratio , random_state=0)
    train_indices, valid_indices = list(ss.split(np.array(labels)[:, np.newaxis], labels))[0]
    trainset = PolicyDatasetC100(trainset)
    trainset,valset = torch.utils.data.Subset(trainset, train_indices), torch.utils.data.Subset(trainset, valid_indices)
    testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.5071, 0.4867, 0.4408],
                                                                [0.2675, 0.2565, 0.2761]),
                                        ]))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True,num_workers=num_worker,
                                              pin_memory=(torch.cuda.is_available()))
    valloader = torch.utils.data.DataLoader(valset, batch_size=train_batch_size, shuffle=True,num_workers=num_worker,
                                              pin_memory=(torch.cuda.is_available()))
    testloader = torch.utils.data.DataLoader(testset, batch_size=test_batch_size, shuffle=False,num_workers=num_worker,
                                             pin_memory=(torch.cuda.is_available()))
    return trainloader,valloader,testloader


