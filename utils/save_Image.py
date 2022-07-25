import os.path
import numpy as np
import torch
from PIL import Image
from torchvision  import transforms as transforms
def change_tensor_to_image(tensor,output_path,name):
    assert tensor.ndim==3 and tensor.shape[0]==3
    tensor=tensor.clone().detach().cpu()
    tensor.mul_(torch.Tensor([0.5071, 0.4867, 0.4408])[:,None,None]).add_(torch.Tensor([0.2675, 0.2565, 0.2761])[:,None,None])
    torch.clip_(tensor,0,1)
    changer=transforms.ToPILImage()
    img=changer(tensor)
    img.save(os.path.join(output_path,name)+".png")