import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets


def load_model(path, model):
    dict = torch.load(path)
    if "student_model" in dict:
        student_ckpt = dict["student_model"]
    else:
        student_ckpt = dict["model"]
    student_ckpt = {k.replace("module.", ""): v for k, v in student_ckpt.items()}
    model.load_state_dict(student_ckpt)
    return model


from models.wrn import wrn_16_2, wrn_40_1, wrn_40_2

model_wo_reviewkd = wrn_16_2(num_classes=100)
model_w_reviewkd = wrn_16_2(num_classes=100)
path_wo_reviewkd = "/checkpoints/wrn40_2_wrn16_2_2/wo_reviewkd_lla.pth"
path_w_reviewkd = "/checkpoints/wrn40_2_wrn16_2_3/w_reviewkd_lla.pth"
model_wo_reviewkd = load_model(path_wo_reviewkd, model_wo_reviewkd)
model_w_reviewkd = load_model(path_w_reviewkd, model_w_reviewkd)

teacher = wrn_40_2(num_classes=100)
teacher_ckpt_path = "/home/sst/product/LLACD/checkpoints/teacher2/wrn_40_2.pth"
teacher = load_model(teacher_ckpt_path, teacher)


from torchvision import transforms

test_dataset = torchvision.datasets.cifar.CIFAR100(
    "/home/sst/dataset/c100",
    train=False,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ]
    ),
)

train_dataset = torchvision.datasets.cifar.CIFAR100(
    "/home/sst/dataset/c100",
    train=True,
    transform=transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]),
        ]
    ),
)
from torch.utils.data import DataLoader

test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=4)
train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=4)


@torch.no_grad()
def test(test_dataloader, model1, model2, teacher):
    model1 = model1.cuda()
    model1.eval()
    model2 = model2.cuda()
    model2.eval()
    teacher = teacher.cuda()
    teacher.eval()

    model1_cls_kl_list = []
    model2_cls_kl_list = []
    model1_acc_list = []
    model2_acc_list = []

    model1_f_mse_list_1 = []
    model1_f_mse_list_2 = []
    model1_f_mse_list_3 = []
    model2_f_mse_list_1 = []
    model2_f_mse_list_2 = []
    model2_f_mse_list_3 = []

    kl_loss = nn.KLDivLoss(reduction="batchmean")
    mse_loss = nn.MSELoss(reduction="mean")
    for i, (image, label) in enumerate(test_dataloader):
        image = image.cuda()
        label = label.cuda()
        features1, output1 = model1(image, is_feat=True)
        features2, output2 = model2(image, is_feat=True)
        featurest, outputt = teacher(image, is_feat=True)
        loss1 = 16 * kl_loss(F.log_softmax(output1 / 4, dim=1), F.softmax(outputt / 4, dim=1))
        model1_cls_kl_list.append(loss1.item())
        loss2 = 16 * kl_loss(F.log_softmax(output2 / 4, dim=1), F.softmax(outputt / 4, dim=1))
        model2_cls_kl_list.append(loss2.item())

        acc1 = (output1.argmax(1) == label).sum() / label.shape[0]
        model1_acc_list.append(acc1.item())
        acc2 = (output2.argmax(1) == label).sum() / label.shape[0]
        model2_acc_list.append(acc2.item())

        model1_f_mse_list_1.append(mse_loss(features1[0], featurest[0]).item())
        model1_f_mse_list_2.append(mse_loss(features1[1], featurest[1]).item())
        model1_f_mse_list_3.append(mse_loss(features1[2], featurest[2]).item())

        model2_f_mse_list_1.append(mse_loss(features2[0], featurest[0]).item())
        model2_f_mse_list_2.append(mse_loss(features2[1], featurest[1]).item())
        model2_f_mse_list_3.append(mse_loss(features2[2], featurest[2]).item())

    import pandas as pd

    result = [
        model1_cls_kl_list,
        model2_cls_kl_list,
        model1_acc_list,
        model2_acc_list,
        model1_f_mse_list_1,
        model2_f_mse_list_1,
        model1_f_mse_list_2,
        model2_f_mse_list_2,
        model1_f_mse_list_3,
        model2_f_mse_list_3,
    ]
    new_result = []
    for a, b, c, d, e, f, g, h, i, j in zip(*result):
        new_result.append([a, b, c, d, e, f, g, h, i, j])
    result = pd.DataFrame(
        new_result,
        columns=[
            "cls_model1",
            "cls_model2",
            "acc_model1",
            "acc_model2",
            "mse_1_model1",
            "mse_1_model2",
            "mse_2_model1",
            "mse_2_model2",
            "mse_3_model1",
            "mse_3_model2",
        ],
    )
    result.to_csv("./distance_train.csv")


test(train_dataloader, model_wo_reviewkd, model_w_reviewkd, teacher)
