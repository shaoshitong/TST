import torch,os
from torchvision import transforms
from PIL import ImageDraw
def detection_vis(tensor, box, label, output_path, name):
    assert tensor.ndim == 3 and tensor.shape[0] == 3

    mean = [123.675/255, 116.28/255, 103.53/255]
    std = [58.395/255, 57.12/255, 57.375/255]
    tensor = tensor.clone().detach().cpu()
    tensor.mul_(torch.Tensor(std)[:, None, None]).add_(
        torch.Tensor(mean)[:, None, None]
    )
    torch.clip_(tensor, 0, 1)
    changer = transforms.ToPILImage()
    img = changer(tensor)
    a = ImageDraw.ImageDraw(img)
    for l,b in zip(label,box):
        min_x, min_y, max_x, max_y = torch.split(
            b, 1, dim=-1)
        min_x,min_y,max_x,max_y = min_x.item() ,min_y.item() ,max_x.item() ,max_y.item()
        a.rectangle((min_x,min_y,max_x,max_y),outline="red",width=5)
        a.text((min_x/2+max_x/2,min_y/2+max_y/2),str(l.item()))
    img.save(os.path.join(output_path, name) + ".png")

import numpy as np

def label_colormap(N=256):

    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    cmap = np.zeros((N, 3))
    for i in range(0, N):
        id = i
        r, g, b = 0, 0, 0
        for j in range(0, 8):
            r = np.bitwise_or(r, (bitget(id, 0) << 7 - j))
            g = np.bitwise_or(g, (bitget(id, 1) << 7 - j))
            b = np.bitwise_or(b, (bitget(id, 2) << 7 - j))
            id = (id >> 3)
        cmap[i, 0] = r
        cmap[i, 1] = g
        cmap[i, 2] = b
    cmap = cmap.astype(np.float32) / 255
    return cmap

def segmentation_vis(tensor, segmentation, output_path, name):
    assert tensor.ndim == 3 and tensor.shape[0] == 3

    mean = [123.675/255, 116.28/255, 103.53/255]
    std = [58.395/255, 57.12/255, 57.375/255]
    tensor = tensor.clone().detach().cpu()
    tensor.mul_(torch.Tensor(std)[:, None, None]).add_(
        torch.Tensor(mean)[:, None, None]
    )
    from mmseg.models import EncoderDecoder
    segmentation = segmentation.clone().detach().cpu()
    labels = torch.from_numpy(label_colormap(torch.max(segmentation).item()+1))
    segmentation = labels[segmentation]
    segmentation = segmentation[0].permute(2,0,1)
    tensor = (tensor*0.01+segmentation*0.99)
    torch.clip_(tensor, 0, 1)
    changer = transforms.ToPILImage()
    img = changer(tensor)
    img.save(os.path.join(output_path, name) + ".png")