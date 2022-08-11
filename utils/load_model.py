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
    print("missing keys:")
    for key in model_state_dict.keys():
        if key not in pretrained_dict:
            print(key)
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(model_state_dict)
    return model


def load_model_from_url(model, url, local_ckpt_path="./checkpoints/teacher2"):
    state_dict = torch.hub.load_state_dict_from_url(
        url,
        model_dir=local_ckpt_path,
        progress=True,
        map_location=torch.device("cpu"),
    )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "model" in state_dict:
        state_dict = state_dict["model"]
    # model.load_state_dict(state_dict)
    model = load_model(state_dict, model)
    return model
