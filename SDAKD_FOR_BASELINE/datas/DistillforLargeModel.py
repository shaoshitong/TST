from timm.data import Mixup
from timm.scheduler.cosine_lr import CosineLRScheduler
from pytorch_optimizer import Shampoo

def mixup():
    mixup = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None,
        prob=1.0, switch_prob=0.5, mode='batch',
        label_smoothing=0.1, num_classes=1000)  # TODO; APPLIED FOR DISTILLATION RESNET50 and SWIN-T
    return mixup


def shampoo(parameters, lr):
    shampoo = Shampoo(parameters, weight_decay=0.000375, momentum=0.9, lr=lr, update_freq=50)
    return shampoo


def cosinescheduler(optimizer, epoch, warmup_lr, warmup_epoch,batch_number):
    cosinescheduler = CosineLRScheduler(optimizer=optimizer,
                                        t_initial=epoch * batch_number,
                                        cycle_mul=1.,
                                        lr_min=warmup_lr * 10,
                                        warmup_lr_init=warmup_lr,
                                        warmup_t=warmup_epoch * batch_number,
                                        cycle_limit=1,
                                        t_in_epochs=False)
    return cosinescheduler

