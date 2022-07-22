import torch
import torch.nn as nn
import torch.nn.functional as F


def load_model(model_dict, model):
    model_state_dict = model.state_dict()
    pretrained_dict = {
        k: v
        for k, v in model_dict.items()
        if k in model_state_dict and v.shape == model_state_dict[k].shape
    }
    print(
        f"the prune number is {round((len(model_state_dict.keys())-len(pretrained_dict.keys()))*100/len(model_state_dict.keys()),3)}%"
    )
    for k, v in pretrained_dict.items():
        if "norm" in k and "num_batches_tracked" not in k:
            pretrained_dict[k].requires_grad = True
        elif "turn_layer" in k and "conv" in k:
            pretrained_dict[k].requires_grad = True
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    return model


def load_model_from_url(model, url):
    state_dict = torch.hub.load_state_dict_from_url(
        url,
        model_dir="/root/.cache/torch/hub/checkpoints",
        progress=True,
        map_location=torch.device("cpu"),
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model = load_model(state_dict, model)
    return model
